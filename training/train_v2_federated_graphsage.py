"""Train V2 GraphSAGE with deterministic, sample-weighted FedAvg.

Requires torch and torch-geometric. Federated aggregation is implemented locally
because Flower is optional and unavailable in the base project runtime.
"""
from __future__ import annotations
import argparse, json, random, sys, time, gc
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/"data/processed/v2"
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))

def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y=np.asarray(y,dtype=np.int64); p=np.asarray(p,dtype=np.float64); pred=p>=0.5
    tp=int(np.sum(pred&(y==1))); fp=int(np.sum(pred&(y==0))); fn=int(np.sum((~pred)&(y==1)))
    order=np.argsort(-p,kind="mergesort"); positives=int(y.sum()); k=max(1,int(np.ceil(.01*len(y))))
    tp_cum=np.cumsum(y[order]); precision=tp_cum/np.arange(1,len(y)+1)
    ap=float((precision*y[order]).sum()/max(1,positives))
    nneg=int((y==0).sum())
    if positives and nneg:
        sorted_idx=np.argsort(p,kind="mergesort"); sorted_p=p[sorted_idx]; ranks=np.empty(len(y),dtype=np.float64)
        start=0
        while start<len(y):
            end=start+1
            while end<len(y) and sorted_p[end]==sorted_p[start]: end+=1
            ranks[sorted_idx[start:end]]=(start+1+end)/2.0; start=end
        auc=float((ranks[y==1].sum()-positives*(positives+1)/2)/(positives*nneg))
    else: auc=float("nan")
    return {"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"f1":2*tp/max(1,2*tp+fp+fn),
            "roc_auc":auc,"pr_auc_average_precision":ap,"recall_at_1_percent":int(tp_cum[k-1])/max(1,positives),
            "recall_at_1_percent_top_k":k,"fraud_count":positives,"fraud_captured_at_1_percent":int(tp_cum[k-1])}

def edge_index_to_mean_csr(edge_index, num_nodes: int, dtype, device):
    """Build target-row/source-column mean adjacency, preserving duplicate-edge weights."""
    import torch
    edge_index=edge_index.to(device=device)
    indices=torch.stack((edge_index[1],edge_index[0]),dim=0)
    values=torch.ones(indices.shape[1],dtype=dtype,device=device)
    coo=torch.sparse_coo_tensor(indices,values,(num_nodes,num_nodes),device=device).coalesce()
    indices=coo.indices(); values=coo.values()
    row=indices[0]
    degree=torch.zeros(num_nodes,dtype=dtype,device=device)
    degree.index_add_(0,row,values)
    mean_values=values/degree.index_select(0,row).clamp_min(1)
    mean_adj=torch.sparse_coo_tensor(indices,mean_values,(num_nodes,num_nodes),device=device).to_sparse_csr()
    del edge_index,indices,values,coo,row,degree,mean_values
    return mean_adj

def sparse_graphsage_forward(model, x, mean_adj):
    """Use sparse mean aggregation with the existing GraphSAGE layer parameters."""
    import torch
    import torch.nn.functional as F
    neighbor=torch.sparse.mm(mean_adj,x)
    x=model.conv1.lin_l(neighbor)+model.conv1.lin_r(x)
    x=F.relu(x)
    x=F.dropout(x,p=model.dropout,training=model.training)
    neighbor=torch.sparse.mm(mean_adj,x)
    x=model.conv2.lin_l(neighbor)+model.conv2.lin_r(x)
    x=F.relu(x)
    x=F.dropout(x,p=model.dropout,training=model.training)
    return model.classifier(x)

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--rounds",type=int,default=5); parser.add_argument("--local-epochs",type=int,default=1); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--lr",type=float,default=0.0005); parser.add_argument("--weight-decay",type=float,default=0.0001); parser.add_argument("--grad-clip",type=float,default=1.0); args=parser.parse_args()
    try:
        import torch
        from models.gnn.graphsage_model import GraphSAGE
    except ImportError as e: raise SystemExit(f"V2 training dependency unavailable: {e}. Install torch and torch-geometric before training.")
    from build_v2_client_graphs import encode, build_edges
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    manifest=json.loads((V2/"feature_manifest.json").read_text(encoding="utf-8")); dim=manifest["feature_dimension"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device":str(device),"cuda_available":torch.cuda.is_available(),
        "gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "input_dim":dim,"federated_rounds":args.rounds,"local_epochs":args.local_epochs}),flush=True)
    client_data=[]
    for cid in (1,2,3):
        with np.load(V2/f"graphs/client_{cid}/graph.npz") as g:
            # from_numpy avoids an extra CPU copy; only the current graph's arrays are staged during loading.
            x_cpu=torch.from_numpy(g["x"]); edge_cpu=torch.from_numpy(g["edge_index"]); y_cpu=torch.from_numpy(g["y"])
            x=x_cpu.to(device); y=y_cpu.to(device)
            adj=edge_index_to_mean_csr(edge_cpu,len(y),x.dtype,device)
            client_data.append((x,adj,y))
            del x_cpu,edge_cpu,y_cpu,x,y,adj
        gc.collect()
    valdf=__import__("pandas").read_csv(V2/"splits/validation.csv",low_memory=False)
    spec={"numeric":manifest["numeric_features"],"categorical_vocab":manifest["categorical_vocab"],"feature_order":manifest["feature_order"]}
    vx=encode(valdf,spec).astype(np.float32,copy=False)
    vx-=np.asarray(manifest["mean"],dtype=np.float32); vx/=np.asarray(manifest["std"],dtype=np.float32)
    ve,_=build_edges(valdf); vy=valdf.isFraud.to_numpy(dtype=np.int64)
    vx=torch.from_numpy(vx).to(device)
    vy_t=torch.from_numpy(vy).to(device)
    val_edge_cpu=torch.from_numpy(ve)
    val_adj=edge_index_to_mean_csr(val_edge_cpu,len(vy),vx.dtype,device)
    del valdf,ve,val_edge_cpu
    def new_model(): return GraphSAGE(input_dim=dim,hidden_dim=128,output_dim=2,dropout=.30).to(device)
    global_model=new_model(); history=[]; best=-1.0; best_state=None
    for rnd in range(args.rounds):
        round_start=time.perf_counter(); client_losses=[]
        local_states=[]; weights=[]
        for client_id,(x,adj,y) in enumerate(client_data,start=1):
            local=new_model(); local.load_state_dict(global_model.state_dict()); local.train()
            opt=torch.optim.Adam(local.parameters(),lr=args.lr,weight_decay=args.weight_decay)
            for _ in range(args.local_epochs):
                opt.zero_grad(set_to_none=True); logits=sparse_graphsage_forward(local,x,adj); loss=torch.nn.functional.cross_entropy(logits,y); loss.backward()
                torch.nn.utils.clip_grad_norm_(local.parameters(),max_norm=args.grad_clip)
                opt.step(); loss_value=float(loss.detach().cpu()); del logits,loss
            client_losses.append({"client":client_id,"train_loss":loss_value})
            print(json.dumps({"round":rnd+1,"client":client_id,"train_loss":loss_value}),flush=True)
            local_states.append({k:v.detach().cpu().clone() for k,v in local.state_dict().items()}); weights.append(len(y))
            del opt,local; gc.collect()
        total=sum(weights); avg={k:sum(state[k]*w for state,w in zip(local_states,weights))/total for k in local_states[0]}
        global_model.load_state_dict(avg); global_model.eval()
        with torch.no_grad():
            val_logits=sparse_graphsage_forward(global_model,vx,val_adj); val_loss=float(torch.nn.functional.cross_entropy(val_logits,vy_t).cpu())
            probs=torch.softmax(val_logits,dim=1)[:,1].cpu().numpy(); del val_logits
        metrics=ranking_metrics(vy,probs); elapsed=time.perf_counter()-round_start
        record={"round":rnd+1,"client_training":client_losses,"aggregation":"sample-count-weighted FedAvg completed","validation_loss":val_loss,"validation_metrics":metrics,"round_runtime_seconds":elapsed}
        history.append(record); print(json.dumps(record),flush=True)
        if metrics["pr_auc_average_precision"]>best: best=metrics["pr_auc_average_precision"]; best_state={k:v.clone() for k,v in global_model.state_dict().items()}
    if best_state is None: raise RuntimeError("No global state produced")
    out=ROOT/"saved_models/v2"; out.mkdir(parents=True,exist_ok=True)
    model=new_model(); model.load_state_dict(best_state); torch.save({"state_dict":best_state,"input_dim":dim,"hidden_dim":128,"output_dim":2,"dropout":.30,"selected_round":int(np.argmax([h["validation"]["pr_auc_average_precision"] for h in history])+1)},out/"global_graphsage_v2.pt")
    (out/"training_metrics.json").write_text(json.dumps({"seed":args.seed,"rounds":args.rounds,"local_epochs":args.local_epochs,"learning_rate":args.lr,"weight_decay":args.weight_decay,"gradient_clip_norm":args.grad_clip,"clients":3,"aggregation":"sample-count-weighted FedAvg","selection_metric":"validation PR-AUC average precision","validation_history":history,"input_dim":dim,"test_used":False,"full_graph_sparse_aggregation":"normalized CSR mean adjacency; no neighbor sampling"},indent=2),encoding="utf-8")
    print(f"Saved frozen V2 global model: {out/'global_graphsage_v2.pt'}")

if __name__=="__main__": main()
