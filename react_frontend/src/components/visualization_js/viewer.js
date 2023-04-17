/**
 * @fileoverview BraxViewer can render static trajectories from json and also
 * connect to a remote brax engine for interactive visualization.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GUI } from 'lil-gui';

import { Animator } from './animator.js';
import { Selector } from './selector.js';
import { createScene, createTrajectory, createSmplTrajectory, createBiomechanicalMesh, createBiomechanicalTrajectory } from './system.js';

function downloadDataUri(name, uri) {
    let link = document.createElement('a');
    link.download = name;
    link.href = uri;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function downloadFile(name, contents, mime) {
    mime = mime || 'text/plain';
    let blob = new Blob([contents], { type: mime });
    let link = document.createElement('a');
    document.body.appendChild(link);
    link.download = name;
    link.href = window.URL.createObjectURL(blob);
    link.onclick = function (e) {
        let scope = this;
        setTimeout(function () {
            window.URL.revokeObjectURL(scope.href);
        }, 1500);
    };
    link.click();
    link.remove();
}

/**
 * Toggles the contact point debug in the scene.
 * @param {!ObjType} obj A Scene or Mesh object.
 * @param {boolean} debug Whether to add contact debugging.
 */
function toggleContactDebug(obj, debug) {
    for (let i = 0; i < obj.children.length; i++) {
        let c = obj.children[i];
        if (c.type == 'AxesHelper') {
            /* toggle visibility on world axis */
            c.visible = debug;
        }
        if (c.type == 'Group' && c.name && c.name.startsWith('contact')) {
            /* toggle visibility of all contact points */
            c.children[0].visible = debug;
        }
        if (c.type == 'Group') {
            /* recurse over group's children */
            for (let j = 0; j < c.children.length; j++) {
                toggleContactDebug(c.children[j], debug);
            }
        }
    }
    if (obj.type == 'Mesh') {
        /* change opacity for each mesh */
        if (debug) {
            obj.material.opacity = 0.6;
            obj.material.transparent = true;
        } else {
            obj.material.opacity = 1.0;
            obj.material.transparent = false;
        }
    }
}

const hoverMaterial = new THREE.MeshPhongMaterial({ color: 0x332722, emissive: 0x114a67 });
const selectMaterial = new THREE.MeshPhongMaterial({
    color: 0x1a1a1a, // A darker color
    //emissive: 0x2194ce, // Add some emissive color to make it stand out
    shininess: 50, // Increase shininess for a more polished look
    opacity: 1.0,
    transparent: false
});
const invisibleMaterial = new THREE.MeshPhongMaterial({ color: 0x000000, opacity: 0.0, transparent: true });

class Viewer {
    constructor(domElement, system, guiElement) {
        // Set +z as pointing up, instead of +y which is the default.
        THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

        this.domElement = domElement;
        this.system = system;
        console.log('system', system)
        this.scene = createScene(system);

        if (system.smpl) {
            this.trajectory = createSmplTrajectory(system, this.scene);
            //this.smplMeshes = this.scene.getObjectById("smpl").children;
        } else if (system.biomechanics) {
            const trajectoryData = system.biomechanics.trajectories;
            this.trajectory = createBiomechanicalTrajectory(trajectoryData, system.dt)
        } else {
            this.trajectory = createTrajectory(system);
        }

        /* set up renderer, camera, and add default scene elements */
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.shadowMap.enabled = true;
        this.renderer.outputEncoding = THREE.sRGBEncoding;

        this.domElement.appendChild(this.renderer.domElement);

        this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
        this.camera.position.set(5, 8, 2);
        this.camera.follow = true;
        this.camera.freezeAngle = false;
        this.camera.followDistance = 10;

        this.scene.background = new THREE.Color(0xa0a0a0);
        this.scene.fog = new THREE.Fog(0xa0a0a0, 40, 60);

        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
        hemiLight.position.set(0, 20, 0);
        this.scene.add(hemiLight);

        const dirLight = new THREE.DirectionalLight(0xffffff);
        dirLight.position.set(3, 10, 10);
        dirLight.castShadow = true;
        dirLight.shadow.camera.top = 10;
        dirLight.shadow.camera.bottom = -10;
        dirLight.shadow.camera.left = -10;
        dirLight.shadow.camera.right = 10;
        dirLight.shadow.camera.near = 0.1;
        dirLight.shadow.camera.far = 40;
        dirLight.shadow.mapSize.width = 4096;   // default is 512
        dirLight.shadow.mapSize.height = 4096;  // default is 512
        this.scene.add(dirLight);
        this.dirLight = dirLight;

        /* set up orbit controls */
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enablePan = false;
        this.controls.enableDamping = true;
        this.controls.addEventListener('start', () => {
            this.setDirty();
        });
        this.controls.addEventListener('change', () => {
            this.setDirty();
        });
        this.controlTargetPos = this.controls.target.clone();

        this.gui = new GUI({ autoPlace: false });

        if (guiElement != undefined) {
            guiElement.appendChild(this.gui.domElement);
        } else {
            // if full screen or in an iframe, this works. 
            this.domElement.parentElement.appendChild(this.gui.domElement);
            this.gui.domElement.style.position = 'absolute';
            this.gui.domElement.style.right = '0%';
            this.gui.domElement.style.top = '0%';
        }

        /* add camera inspectors */
        const cameraFolder = this.gui.addFolder('Camera');
        cameraFolder.add(this.camera, 'freezeAngle').name('Freeze Angle');
        cameraFolder.add(this.camera, 'follow').name('Follow Target');
        cameraFolder.add(this.camera, 'followDistance')
            .name('Follow Distance')
            .min(1)
            .max(50);
        cameraFolder.close()

        /* set up animator and load trajectory */
        this.animator = new Animator(this);
        if (this.trajectory) {
            console.log('trajectory', this.trajectory)
            this.animator.load(this.trajectory, {});
        }

        /* add body inspectors */
        const bodiesFolder = this.gui.addFolder('Bodies');
        bodiesFolder.close();

        this.bodyFolders = {};

        for (let c of this.scene.children) {
            if (!c.name || c.name.startsWith('contact')) continue;
            const folder = bodiesFolder.addFolder(c.name);
            this.bodyFolders[c.name] = folder;
            folder.close();

            function defaults() {
                for (const gui of arguments) {
                    gui.step(0.01).listen().domElement.style.pointerEvents = 'none';
                }
            }
            defaults(
                folder.add(c.position, 'x').name('pos.x'),
                folder.add(c.position, 'y').name('pos.y'),
                folder.add(c.position, 'z').name('pos.z'),
                folder.add(c.rotation, 'x').name('rot.x'),
                folder.add(c.rotation, 'y').name('rot.y'),
                folder.add(c.rotation, 'z').name('rot.z'),
            );
        }
        // let saveFolder = this.gui.addFolder('Save / Capture');
        // saveFolder.add(this, 'saveScene').name('Save Scene');
        // saveFolder.add(this, 'saveImage').name('Capture Image');
        // saveFolder.close();

        /* debugger */
        // this.contactDebug = system.states.contact !== null;
        // let debugFolder = this.gui.addFolder('Debugger');
        // debugFolder.add(this, 'contactDebug')
        //     .name(system.states.contact ? 'contacts' : 'axis')
        //     .onChange((value) => this.setContactDebug(value));

        /* done setting up the gui */
        //this.gui.close();

        /* set up body selector */
        this.selector = new Selector(this);
        this.selector.addEventListener(
            'hoveron', (evt) => this.setHover(evt.object, true));
        this.selector.addEventListener(
            'hoveroff', (evt) => this.setHover(evt.object, false));
        this.selector.addEventListener(
            'select', (evt) => this.setSelected(evt.object, true));
        this.selector.addEventListener(
            'deselect', (evt) => this.setSelected(evt.object, false));

        if (system.biomechanics) {
            console.log('setting up follower')
            const biomechanicsGroup = this.scene.getObjectByName('biomechanics')
            this.target = biomechanicsGroup.children[0];
        } else if (system.smpl) {
            // TODO: this code doesn't actually work because the morphTragets use
            // absolute positions but the "position" of the mesh element doesn't
            // get updated. To get this working, we will need to extract an average
            // position from the morphTargets and use that as the position of the
            // mesh element.
            if (system.smpl.ids.length == 1) {
                console.log('setting up SMPL follower')
                const smplGroup = this.scene.getObjectByName('smpl')
                this.target = smplGroup.children[0];
                console.log('target', this.target)
            }
        } else {
            this.defaultTarget = this.selector.selectable[0];
            this.target = this.defaultTarget;
        }

        /* get ready to render first frame */
        this.setDirty();

        window.onload = (evt) => this.setSize();
        window.addEventListener('resize', (evt) => this.setSize(), false);
        requestAnimationFrame(() => this.setSize());

        const resizeObserver =
            new ResizeObserver(() => this.resizeCanvasToDisplaySize());
        resizeObserver.observe(this.domElement, { box: 'content-box' });

        /* start animation */
        this.animate();

        this.smpl_frames = 0;
    }

    setFilter(filter) {
        /* if true, then only show the meshes that are in the selected list and make the others 
        transparent */

        this.filter = filter;
        const selected = this.selector.selected;

        // fetch the meshes from the "smpl" group in the scene
        const meshes = this.scene.getObjectByName('smpl').children;

        meshes.forEach((mesh) => {
            if (filter) {
                // now see if object is in the list
                if (selected.includes(mesh)) {
                    mesh.material = selectMaterial;
                } else {
                    mesh.material = invisibleMaterial;
                }
            } else {
                // default behavior for being selected
                mesh.material = selected.includes(mesh) ? selectMaterial : mesh.baseMaterial;
            }
        });
    }

    setDirty() {
        this.needsRender = true;
    }

    setSize(w, h) {
        if (w === undefined) {
            w = this.domElement.offsetWidth;
        }
        if (h === undefined) {
            h = this.domElement.clientHeight;
        }
        if (this.camera.type == 'OrthographicCamera') {
            this.camera.right =
                this.camera.left + w * (this.camera.top - this.camera.bottom) / h;
        } else {
            this.camera.aspect = w / h;
        }
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        this.setDirty();
    }

    resizeCanvasToDisplaySize() {
        // look up canvas size
        const width = this.domElement.clientWidth;
        const height = this.domElement.clientHeight;
        this.setSize(width, height);
    }

    render() {
        //toggleContactDebug(this.scene, this.contactDebug);
        this.renderer.render(this.scene, this.camera);
        this.needsRender = false;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.animator.update();

        // make sure the orbiter is pointed at the right target
        const targetPos = new THREE.Vector3();
        this.target.getWorldPosition(targetPos);

        // if the target gets too far from the camera, nudge the camera
        if (this.camera.follow) {
            this.controls.target.lerp(targetPos, 0.1);
            if (this.camera.position.distanceTo(this.controls.target) >
                this.camera.followDistance) {
                const followBehind = this.controls.target.clone()
                    .sub(this.camera.position)
                    .normalize()
                    .multiplyScalar(this.camera.followDistance)
                    .sub(this.controls.target)
                    .negate();
                this.camera.position.lerp(followBehind, 0.5);
                this.setDirty();
            }
        }

        // make sure target stays within shadow map region
        this.dirLight.position.set(
            targetPos.x + 3, targetPos.y + 10, targetPos.z + 10);
        this.dirLight.target = this.target;

        if (this.controls.update()) {
            this.setDirty();
        }

        // if freezeAngle requested, move the camera on xz plane to match target
        if (this.camera.freezeAngle) {
            const off = new THREE.Vector3();
            off.add(this.controls.target).sub(this.controlTargetPos);
            off.setComponent(1, 0);
            if (off.lengthSq() > 0) {
                this.camera.position.add(off);
                this.setDirty();
            }
        }
        this.controlTargetPos.copy(this.controls.target);

        if (this.needsRender) {
            this.render();
        }
    }

    saveImage() {
        this.render();
        const imageData = this.renderer.domElement.toDataURL();
        downloadDataUri('markerlessmocap.png', imageData);
    }

    saveScene() {
        downloadFile('system.json', JSON.stringify(this.system));
    }

    setContactDebug(val) {
        this.contactDebug = val;
    }

    setHover(object, hovering) {
        this.setDirty();
        if (!object.selected) {
            object.traverse(function (child) {
                if (child instanceof THREE.Mesh) {
                    child.material = hovering ? hoverMaterial : child.baseMaterial;
                }
            });
        }
        if (object.name in this.bodyFolders) {
            const titleElement =
                this.bodyFolders[object.name].domElement.querySelector('.title');
            if (titleElement) {
                titleElement.style.backgroundColor = hovering ? '#2fa1d6' : '#000';
            }
        }
    }

    setSelected(object, selected) {
        console.log("setSelected: " + object.name + " " + selected)
        object.selected = selected;
        this.target = selected ? object : this.defaultTarget;
        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                child.material = selected ? selectMaterial : child.baseMaterial;
                child.castShadow = selected ? true : false;
            }
        });
        if (object.name in this.bodyFolders) {
            if (object.selected) {
                this.bodyFolders[object.name].open();
            } else {
                this.bodyFolders[object.name].close();
            }
        }
        this.setDirty();
    }

    getSelectedIds() {
        const names = this.selector.selected.map((object) => object.name);
        // for names parse the numeric part after person_
        const ids = names.map((name) => parseInt(name.split('_')[1]));
        return ids;
    }

    close() {
        // Remove the canvas from the DOM
        this.domElement.removeChild(this.renderer.domElement);

        // Remove the GUI from the DOM
        this.gui.domElement.parentElement.removeChild(this.gui.domElement);
    }
}

export { Viewer };