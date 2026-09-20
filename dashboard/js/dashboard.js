/* ================================================================
   RFGN — REINFORCED FEDERATED GRAPH NETWORK
   HOMEPAGE JAVASCRIPT
   ================================================================ */


/* ================================================================
   01 — GLOBAL CONFIGURATION
   ================================================================ */

const RFGN_CONFIG = {

    matrix: {

        characters:
            "01RFGN<>[]{}",

        fontSize:
            14,

        opacity:
            0.46,

        minSpeed:
            0.35,

        maxSpeed:
            1.10

    },

    animation: {

        reducedMotion:
            window.matchMedia(
                "(prefers-reduced-motion: reduce)"
            ).matches

    }

};


/* ================================================================
   02 — DOM READY
   ================================================================ */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        initializeMatrixRain();

        initializeNavigation();

        initializeHeroButtons();

        initializeScrollNavigation();

        initializeMobileMenu();

        initializeLogoErrorHandling();

    }
);


/* ================================================================
   03 — MATRIX RAIN
   ================================================================ */

function initializeMatrixRain() {

    const canvas =
        document.getElementById(
            "matrixCanvas"
        );


    if (!canvas) {

        console.warn(
            "RFGN: Matrix canvas not found."
        );

        return;

    }


    const context =
        canvas.getContext(
            "2d"
        );


    if (!context) {

        console.warn(
            "RFGN: Canvas context unavailable."
        );

        return;

    }


    let width = 0;

    let height = 0;

    let columns = 0;

    let drops = [];

    let speeds = [];

    let lastFrameTime = 0;


    const fontSize =
        RFGN_CONFIG.matrix.fontSize;


    const characters =
        RFGN_CONFIG.matrix.characters;


    /*
     * Resize the canvas according to the
     * current viewport.
     */

    function resizeCanvas() {

        const devicePixelRatio =
            Math.min(
                window.devicePixelRatio || 1,
                2
            );


        width =
            window.innerWidth;

        height =
            window.innerHeight;


        canvas.width =
            Math.floor(
                width *
                devicePixelRatio
            );


        canvas.height =
            Math.floor(
                height *
                devicePixelRatio
            );


        canvas.style.width =
            `${width}px`;


        canvas.style.height =
            `${height}px`;


        context.setTransform(
            devicePixelRatio,
            0,
            0,
            devicePixelRatio,
            0,
            0
        );


        columns =
            Math.ceil(
                width /
                fontSize
            );


        drops =
            new Array(
                columns
            );


        speeds =
            new Array(
                columns
            );


        for (
            let i = 0;
            i < columns;
            i++
        ) {

            /*
             * Start many streams above
             * the visible viewport.
             */

            drops[i] =
                Math.random() *
                -100;


            speeds[i] =
                RFGN_CONFIG.matrix.minSpeed +
                Math.random() *
                (
                    RFGN_CONFIG.matrix.maxSpeed -
                    RFGN_CONFIG.matrix.minSpeed
                );

        }


        context.font =
            `${fontSize}px monospace`;

    }


    /*
     * Draw one Matrix frame.
     */

    function drawMatrix(
        timestamp
    ) {

        if (
            !lastFrameTime
        ) {

            lastFrameTime =
                timestamp;

        }


        const delta =
            Math.min(
                timestamp -
                lastFrameTime,

                50
            );


        lastFrameTime =
            timestamp;


        /*
         * Convert elapsed time into
         * a stable animation multiplier.
         */

        const timeScale =
            delta /
            16.67;


        /*
         * Fade the previous frame.
         */

        context.fillStyle =
            "rgba(0, 0, 0, 0.065)";


        context.fillRect(
            0,
            0,
            width,
            height
        );


        context.font =
            `${fontSize}px monospace`;


        /*
         * Draw every Matrix column.
         */

        for (
            let i = 0;
            i < columns;
            i++
        ) {

            const randomIndex =
                Math.floor(
                    Math.random() *
                    characters.length
                );


            const character =
                characters[
                    randomIndex
                ];


            const x =
                i *
                fontSize;


            const y =
                drops[i] *
                fontSize;


            /*
             * A small percentage of
             * characters become brighter.
             */

            const bright =
                Math.random() >
                0.94;


            if (bright) {

                context.fillStyle =
                    "rgba(30, 255, 135, 0.88)";

            } else {

                context.fillStyle =
                    "rgba(0, 210, 95, 0.38)";

            }


            context.fillText(
                character,
                x,
                y
            );


            /*
             * Move the stream downward.
             */

            drops[i] +=
                speeds[i] *
                timeScale;


            /*
             * Randomly restart streams
             * once they leave the viewport.
             */

            if (
                y >
                height
            ) {

                if (
                    Math.random() >
                    0.975
                ) {

                    drops[i] =
                        Math.random() *
                        -35;

                }

            }

        }


        /*
         * Continue animation.
         */

        requestAnimationFrame(
            drawMatrix
        );

    }


    /*
     * Reduced-motion users get a
     * static subtle Matrix background.
     */

    function drawStaticMatrix() {

        context.clearRect(
            0,
            0,
            width,
            height
        );


        context.font =
            `${fontSize}px monospace`;


        for (
            let i = 0;
            i < columns;
            i++
        ) {

            const character =
                characters[
                    Math.floor(
                        Math.random() *
                        characters.length
                    )
                ];


            const x =
                i *
                fontSize;


            const y =
                Math.random() *
                height;


            context.fillStyle =
                "rgba(0, 210, 95, 0.18)";


            context.fillText(
                character,
                x,
                y
            );

        }

    }


    resizeCanvas();


    window.addEventListener(
        "resize",
        () => {

            resizeCanvas();

            if (
                RFGN_CONFIG.animation
                    .reducedMotion
            ) {

                drawStaticMatrix();

            }

        }
    );


    if (
        RFGN_CONFIG.animation
            .reducedMotion
    ) {

        drawStaticMatrix();

    } else {

        requestAnimationFrame(
            drawMatrix
        );

    }

}


/* ================================================================
   04 — NAVIGATION
   ================================================================ */

function initializeNavigation() {

    const navLinks =
        document.querySelectorAll(
            ".nav-link"
        );


    if (!navLinks.length) {

        return;

    }


    navLinks.forEach(
        link => {

            link.addEventListener(
                "click",
                event => {

                    const target =
                        link.getAttribute(
                            "href"
                        );


                    /*
                     * Only handle internal
                     * section navigation.
                     */

                    if (
                        !target ||
                        !target.startsWith("#")
                    ) {

                        return;

                    }


                    const targetElement =
                        document.querySelector(
                            target
                        );


                    if (!targetElement) {

                        return;

                    }


                    event.preventDefault();


                    /*
                     * Smooth scrolling.
                     */

                    targetElement.scrollIntoView(
                        {
                            behavior:
                                RFGN_CONFIG
                                    .animation
                                    .reducedMotion
                                    ? "auto"
                                    : "smooth",

                            block:
                                "start"
                        }
                    );


                    setActiveNavigation(
                        target
                    );

                }
            );

        }
    );

}


/* ================================================================
   05 — ACTIVE NAVIGATION
   ================================================================ */

function setActiveNavigation(
    target
) {

    const navLinks =
        document.querySelectorAll(
            ".nav-link"
        );


    navLinks.forEach(
        link => {

            link.classList.remove(
                "active"
            );


            if (
                link.getAttribute(
                    "href"
                ) === target
            ) {

                link.classList.add(
                    "active"
                );

            }

        }
    );

}


/* ================================================================
   06 — SCROLL NAVIGATION
   ================================================================ */

function initializeScrollNavigation() {

    const sections =
        document.querySelectorAll(
            "section[id]"
        );


    if (!sections.length) {

        return;

    }


    /*
     * IntersectionObserver lets us
     * determine which section is
     * currently visible.
     */

    const observer =
        new IntersectionObserver(
            entries => {

                /*
                 * Find the section with
                 * the strongest visibility.
                 */

                const visibleSections =
                    entries
                        .filter(
                            entry =>
                                entry.isIntersecting
                        )
                        .sort(
                            (
                                a,
                                b
                            ) =>
                                b.intersectionRatio -
                                a.intersectionRatio
                        );


                if (
                    !visibleSections.length
                ) {

                    return;

                }


                const currentSection =
                    visibleSections[0]
                        .target;


                const id =
                    currentSection.id;


                setActiveNavigation(
                    `#${id}`
                );

            },
            {

                threshold:
                    [
                        0.25,
                        0.5,
                        0.75
                    ],

                rootMargin:
                    "-10% 0px -35% 0px"

            }
        );


    sections.forEach(
        section =>
            observer.observe(
                section
            )
    );

}


/* ================================================================
   07 — HERO BUTTONS
   ================================================================ */

function initializeHeroButtons() {

    const exploreButton =
        document.getElementById(
            "exploreButton"
        );


    const watchButton =
        document.getElementById(
            "watchButton"
        );


    /*
     * Explore Platform
     */

    if (
        exploreButton
    ) {

        exploreButton.addEventListener(
            "click",
            () => {

                /*
                 * Login page will be created
                 * next.
                 */

                window.location.href =
                    "login.html";

            }
        );

    }


    /*
     * Watch Overview
     */

    if (
        watchButton
    ) {

        watchButton.addEventListener(
            "click",
            () => {

                const aboutSection =
                    document.getElementById(
                        "about"
                    );


                if (
                    aboutSection
                ) {

                    aboutSection.scrollIntoView(
                        {
                            behavior:
                                RFGN_CONFIG
                                    .animation
                                    .reducedMotion
                                    ? "auto"
                                    : "smooth",

                            block:
                                "start"
                        }
                    );

                }

            }
        );

    }

}


/* ================================================================
   08 — MOBILE MENU
   ================================================================ */

function initializeMobileMenu() {

    const menuButton =
        document.getElementById(
            "mobileMenu"
        );


    if (
        !menuButton
    ) {

        return;

    }


    /*
     * Current homepage has a simple
     * mobile fallback.
     *
     * We will replace this with a
     * proper mobile navigation panel
     * when the complete four-page
     * interface is assembled.
     */

    menuButton.addEventListener(
        "click",
        () => {

            createMobileNavigation();

        }
    );

}


/* ================================================================
   09 — MOBILE NAVIGATION PANEL
   ================================================================ */

function createMobileNavigation() {

    /*
     * Prevent duplicate menu creation.
     */

    const existingMenu =
        document.getElementById(
            "rfgnMobileNavigation"
        );


    if (
        existingMenu
    ) {

        existingMenu.remove();

        document.body.style.overflow =
            "";

        return;

    }


    const menu =
        document.createElement(
            "div"
        );


    menu.id =
        "rfgnMobileNavigation";


    /*
     * Inline styles keep this temporary
     * mobile panel independent from the
     * desktop layout.
     */

    menu.style.position =
        "fixed";

    menu.style.top =
        "96px";

    menu.style.left =
        "18px";

    menu.style.right =
        "18px";

    menu.style.zIndex =
        "999";

    menu.style.padding =
        "25px";

    menu.style.background =
        "rgba(3, 15, 25, 0.96)";

    menu.style.border =
        "1px solid rgba(50, 184, 255, 0.42)";

    menu.style.backdropFilter =
        "blur(18px)";

    menu.style.webkitBackdropFilter =
        "blur(18px)";

    menu.style.boxShadow =
        "0 20px 50px rgba(0,0,0,0.55)";


    const links =
        [
            {
                label:
                    "HOME",

                target:
                    "#home"
            },

            {
                label:
                    "ABOUT RFGN",

                target:
                    "#about"
            },

            {
                label:
                    "TECHNOLOGY",

                target:
                    "#technology"
            },

            {
                label:
                    "ARCHITECTURE",

                target:
                    "#architecture"
            },

            {
                label:
                    "FEATURES",

                target:
                    "#features"
            },

            {
                label:
                    "CONTACT",

                target:
                    "#contact"
            }
        ];


    links.forEach(
        item => {

            const link =
                document.createElement(
                    "a"
                );


            link.href =
                item.target;


            link.textContent =
                item.label;


            link.style.display =
                "block";


            link.style.padding =
                "15px 10px";


            link.style.color =
                "rgba(238,246,251,0.88)";


            link.style.fontSize =
                "14px";


            link.style.letterSpacing =
                "1.5px";


            link.style.borderBottom =
                "1px solid rgba(50,170,230,0.10)";


            link.addEventListener(
                "click",
                () => {

                    menu.remove();

                    document.body.style.overflow =
                        "";

                }
            );


            menu.appendChild(
                link
            );

        }
    );


    document.body.appendChild(
        menu
    );


    /*
     * Prevent background scrolling
     * while menu is open.
     */

    document.body.style.overflow =
        "hidden";

}


/* ================================================================
   10 — LOGO ERROR HANDLING
   ================================================================ */

function initializeLogoErrorHandling() {

    const logos =
        document.querySelectorAll(
            "img[src*='RFGN_logo_reference']"
        );


    if (!logos.length) {

        return;

    }


    logos.forEach(
        logo => {

            logo.addEventListener(
                "error",
                () => {

                    console.error(
                        "RFGN: Logo image could not be loaded."
                    );


                    /*
                     * Keep the image area
                     * but reduce visual disruption.
                     */

                    logo.style.opacity =
                        "0.15";

                }
            );

        }
    );

}


/* ================================================================
   11 — ESCAPE KEY
   ================================================================ */

document.addEventListener(
    "keydown",
    event => {

        if (
            event.key !==
            "Escape"
        ) {

            return;

        }


        const mobileMenu =
            document.getElementById(
                "rfgnMobileNavigation"
            );


        if (
            mobileMenu
        ) {

            mobileMenu.remove();

            document.body.style.overflow =
                "";

        }

    }
);


/* ================================================================
   12 — WINDOW VISIBILITY
   ================================================================ */

document.addEventListener(
    "visibilitychange",
    () => {

        /*
         * The browser automatically throttles
         * requestAnimationFrame in background
         * tabs. This handler simply resets
         * timing after returning to the page
         * so the Matrix animation doesn't jump.
         */

        if (
            document.visibilityState ===
            "visible"
        ) {

            /*
             * No additional animation loop
             * is created here.
             *
             * requestAnimationFrame handles
             * continuation automatically.
             */

        }

    }
);


/* ================================================================
   13 — CONSOLE BRANDING
   ================================================================ */

console.log(
    "%cRFGN",
    "font-size: 28px; font-weight: 700; color: #20c8ff;"
);


console.log(
    "%cReinforced Federated Graph Network",
    "font-size: 13px; color: #8aa8bb;"
);


console.log(
    "%cIntelligence Beyond Connections",
    "font-size: 12px; color: #168cff;"
);


/* ================================================================
   END OF RFGN HOMEPAGE JAVASCRIPT
   ================================================================ */