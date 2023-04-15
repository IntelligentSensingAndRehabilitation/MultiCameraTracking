import * as THREE from 'three';

function createCheckerBoard() {
    const width = 2;
    const height = 2;

    const size = width * height;
    const data = new Uint8Array(4 * size);
    const colors = [new THREE.Color(0x999999), new THREE.Color(0x888888)];

    for (let i = 0; i < size; i++) {
        const stride = i * 4;
        const ck = [0, 1, 1, 0];
        const color = colors[ck[i]];
        data[stride + 0] = Math.floor(color.r * 255);
        data[stride + 1] = Math.floor(color.g * 255);
        data[stride + 2] = Math.floor(color.b * 255);
        data[stride + 3] = 255;
    }
    const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(1000, 1000);
    texture.needsUpdate = true;
    return new THREE.MeshPhongMaterial({ map: texture });
}

function getCapsuleAxisSize(capsule) {
    return capsule.length * 2;
}

function getSphereAxisSize(sphere) {
    return sphere.radius * 2;
}

function getBoxAxisSize(box) {
    return Math.max(box.halfsize[0], box.halfsize[1], box.halfsize[2]) * 4;
}

/**
 * Gets an axis size for a mesh.
 * @param {!ObjType} geom a geometry object
 * @returns {!float} the axis size
 */
function getMeshAxisSize(geom) {
    let size = 0;
    for (let i = 0; i < geom.vert.length; i++) {
        let v = geom.vert[i];
        size = Math.max(v[0], v[1], v[2], size);
    }
    return size * 2;
}

function createCapsule(capsule, mat) {
    const sphere_geom = new THREE.SphereGeometry(capsule.radius, 16, 16);
    const cylinder_geom = new THREE.CylinderGeometry(
        capsule.radius, capsule.radius, capsule.length);

    const sphere1 = new THREE.Mesh(sphere_geom, mat);
    sphere1.baseMaterial = sphere1.material;
    sphere1.position.set(0, 0, capsule.length / 2);
    sphere1.castShadow = true;
    sphere1.layers.enable(1);

    const sphere2 = new THREE.Mesh(sphere_geom, mat);
    sphere2.baseMaterial = sphere2.material;
    sphere2.position.set(0, 0, -capsule.length / 2);
    sphere2.castShadow = true;
    sphere2.layers.enable(1);

    const cylinder = new THREE.Mesh(cylinder_geom, mat);
    cylinder.baseMaterial = cylinder.material;
    cylinder.castShadow = true;
    cylinder.rotation.x = -Math.PI / 2;
    cylinder.layers.enable(1);

    const group = new THREE.Group();
    group.add(sphere1, sphere2, cylinder);
    return group;
}

function createBox(box, mat) {
    const geom = new THREE.BoxBufferGeometry(
        2 * box.halfsize[0], 2 * box.halfsize[1], 2 * box.halfsize[2]);
    const mesh = new THREE.Mesh(geom, mat);
    mesh.castShadow = true;
    mesh.baseMaterial = mesh.material;
    mesh.layers.enable(1);
    return mesh;
}

function createPlane(plane, mat) {
    const geometry = new THREE.PlaneGeometry(2000, 2000);
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.receiveShadow = true;
    mesh.baseMaterial = mesh.material;

    return mesh;
}

function createSphere(sphere, mat) {
    const geom = new THREE.SphereGeometry(sphere.radius, 16, 16);
    const mesh = new THREE.Mesh(geom, mat);
    mesh.castShadow = true;
    mesh.baseMaterial = mesh.material;
    mesh.layers.enable(1);
    return mesh;
}

function createMesh(meshGeom, mat) {
    const bufferGeometry = new THREE.BufferGeometry();
    const vertices = meshGeom.vert;
    const positions = new Float32Array(vertices.length * 3);
    // Convert the coordinate system.
    vertices.forEach(function (vertice, i) {
        positions[i * 3] = vertice[0] * 0;
        positions[i * 3 + 1] = vertice[1] * 0;
        positions[i * 3 + 2] = vertice[2] * 0;
    });
    const indices = new Uint16Array(meshGeom.face.flat());
    bufferGeometry.setAttribute(
        'position', new THREE.BufferAttribute(positions, 3));
    bufferGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
    bufferGeometry.computeVertexNormals();

    const mesh = new THREE.Mesh(bufferGeometry, mat);
    mesh.castShadow = true;
    mesh.baseMaterial = mesh.material;
    mesh.layers.enable(1);
    return mesh;
}

function createScene(system) {
    const scene = new THREE.Scene();

    const meshGeoms = {};
    if (system.meshes) {
        Object.entries(system.meshes).forEach(function (geom) {
            meshGeoms[geom[0]] = geom[1];
        });
    }

    // Add a world axis for debugging.
    const worldAxis = new THREE.AxesHelper(100);
    worldAxis.visible = false;
    scene.add(worldAxis);

    const ground = createPlane(null, createCheckerBoard());
    scene.add(ground);

    let minAxisSize = 1e6;
    Object.entries(system.geoms).forEach(function (geom) {
        const name = geom[0];
        const parent = new THREE.Group();
        parent.name = name.replaceAll('/', '_');  // sanitize node name
        geom[1].forEach(function (collider) {
            const rgba = collider.rgba;
            const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
            const mat = (collider.name == 'Plane') ?
                createCheckerBoard() :
                (collider.name == 'heightMap') ?
                    new THREE.MeshStandardMaterial({ color: color, flatShading: true }) :
                    new THREE.MeshPhongMaterial({ color: color });
            let child;
            let axisSize;
            if (collider.name == 'Box') {
                child = createBox(collider, mat);
                axisSize = getBoxAxisSize(collider);
            } else if (collider.name == 'Capsule') {
                child = createCapsule(collider, mat);
                axisSize = getCapsuleAxisSize(collider);
            } else if (collider.name == 'Plane') {
                child = createPlane(collider.plane, mat);
            } else if (collider.name == 'Sphere') {
                child = createSphere(collider, mat);
                axisSize = getSphereAxisSize(collider);
            } else if (collider.name == 'HeightMap') {
                console.log('heightMap not implemented');
                return;
            } else if (collider.name == 'Mesh') {
                child = createMesh(collider, mat);
                axisSize = getMeshAxisSize(collider);
            } else if ('clippedPlane' in collider) {
                console.log('clippedPlane not implemented');
                return;
            } else if (collider.name == 'Convex') {
                console.log('convex not implemented');
                return;
            }
            if (collider.transform.rot) {
                child.quaternion.set(
                    collider.transform.rot[1], collider.transform.rot[2],
                    collider.transform.rot[3], collider.transform.rot[0]);
            }
            if (collider.transform.pos) {
                child.position.set(
                    collider.transform.pos[0], collider.transform.pos[1],
                    collider.transform.pos[2]);
            }
            if (axisSize) {
                const debugAxis = new THREE.AxesHelper(axisSize);
                debugAxis.visible = false;
                child.add(debugAxis);
                minAxisSize = Math.min(minAxisSize, axisSize);
            }
            parent.add(child);
        });
        scene.add(parent);
    });

    if (system.smpl) {
        const faces = system.smpl.faces;
        const frameData = system.smpl.meshes;

        var smplMeshes = []
        var smplKeyframeTracks = []

        frameData.forEach((frame, index) => {
            const res = appendSmplFrame(frame, smplMeshes, smplKeyframeTracks, scene, index, faces);
            smplMeshes = res.meshes
            smplKeyframeTracks = res.tracks
        });
    }

    if (system.states.contact) {
        /* add contact point spheres  */
        for (let i = 0; i < system.states.contact.pos[0].length; i++) {
            const parent = new THREE.Group();
            parent.name = 'contact' + i;
            let child;

            const mat = new THREE.MeshPhongMaterial({ color: 0xff0000 });
            const sphere_geom = new THREE.SphereGeometry(minAxisSize / 20.0, 6, 6);
            child = new THREE.Mesh(sphere_geom, mat);
            child.baseMaterial = child.material;
            child.castShadow = false;
            child.position.set(0, 0, 0);

            parent.add(child);
            scene.add(parent);
        }
    }

    if (system.keypoints) {
        /* add keypoint spheres  */
        for (let i = 0; i < system.keypoints[0].length; i++) {
            const parent = new THREE.Group();
            parent.name = 'keypoint' + i;
            let child;

            const mat = new THREE.MeshPhongMaterial({ color: 0xff00ff });
            const sphere_geom = new THREE.SphereGeometry(0.025, 12, 12);
            child = new THREE.Mesh(sphere_geom, mat);
            child.baseMaterial = child.material;
            child.castShadow = false;
            child.position.set(0, 0, 0);

            parent.add(child);
            scene.add(parent);
        }
    }

    return scene;
}

function createTrajectory(system) {
    const times =
        [...Array(system.states.x.pos.length).keys()].map(x => x * system.dt);
    const tracks = [];

    Object.entries(system.geoms).forEach(function (geom_tuple) {
        const name = geom_tuple[0];
        const geom = geom_tuple[1];
        const i = geom[0].link_idx;
        if (i == null) {
            return;
        }
        const group = name.replaceAll('/', '_');  // sanitize node name
        const pos = system.states.x.pos.map(p => [p[i][0], p[i][1], p[i][2]]);
        const rot =
            system.states.x.rot.map(r => [r[i][1], r[i][2], r[i][3], r[i][0]]);
        tracks.push(new THREE.VectorKeyframeTrack(
            'scene/' + group + '.position', times, pos.flat()));
        tracks.push(new THREE.QuaternionKeyframeTrack(
            'scene/' + group + '.quaternion', times, rot.flat()));
    });

    if (system.states.contact) {
        /* add contact debug point trajectory */
        for (let i = 0; i < system.states.contact.pos[0].length; i++) {
            const group = 'contact' + i;
            const pos = system.states.contact.pos.map(p => [p[i][0], p[i][1], p[i][2]]);
            const visible = system.states.contact.penetration.map(p => p[i] > 1e-6);
            tracks.push(new THREE.VectorKeyframeTrack(
                'scene/' + group + '.position', times, pos.flat(),
                THREE.InterpolateDiscrete));
            tracks.push(new THREE.BooleanKeyframeTrack(
                'scene/' + group + '.visible', times, visible,
                THREE.InterpolateDiscrete));
        }
    }

    return new THREE.AnimationClip('Action', -1, tracks);
}

function createKeypointTrajectory(system) {
    const times =
        [...Array(system.keypoints.length).keys()].map((x) => x * system.dt);
    const tracks = [];

    console.log("length of keypoints: " + system.keypoints.length);

    // Assuming the `system.keypoints` has the structure [time][joint][coordinate]
    for (let jointIndex = 0; jointIndex < system.keypoints[0].length; jointIndex++) {
        const jointPositions = system.keypoints.map((time) => time[jointIndex]);
        tracks.push(
            new THREE.VectorKeyframeTrack(
                "scene/keypoint" + jointIndex + ".position",
                times,
                jointPositions.flat()
            )
        );
    }

    return new THREE.AnimationClip("Action", -1, tracks);
}


function createVisibilityKeyframeTrack(name, showIdx) {
    const dt = 5 / 30;

    const times = [showIdx - 0.01, showIdx].map(t => t * dt);
    const values = [false, true, true, false];

    console.log(name, times, values);

    const track = new THREE.BooleanKeyframeTrack(
        `scene/${name}.visible`,
        times,
        values
    );

    return track;
}

function createInvisibilityKeyframeTrack(name, showIdx, hideIdx) {
    const dt = 5 / 30;

    const times = [showIdx - 0.01, showIdx, hideIdx - 1, hideIdx - 1 + 0.01].map(t => t * dt);
    const values = [false, true, true, false];

    console.log(name, times, values);

    const track = new THREE.BooleanKeyframeTrack(
        `scene/${name}.visible`,
        times,
        values
    );

    return track;
}

function appendSmplFrame(frameData, smplMeshes, keyframeTracks, scene, timeIndex, faces) {
    /*
    frameData contains the data for a single frame of the SMPL animation. it is a list of dictionaries,
    where each dictionary element has an ID field with the numerical identity for that person and a verts
    field with the vertices for that person for the specified frame. This is used to update teh smplMeshes
    data.

    smplMeshes is a list of THREE.Mesh objects, one for each person in the scene. The order of the meshes
    in the list should match the numeric IDs for the people in the frameData list.

    faces are the face vertex ids used when creating new meshes

    scene is the scene object to which the meshes should be added.

    timeIndex is only used when creating new meshes to add the appropriate invisible frames when people
    were not visible in earlier frames. for any frames wehre an ID is not in the frameData list, we need
    to set the visible user data to false so the animation layer will not show the mesh for that frame.
    */


    // loop through all the meshes and if there is not a personId that matches
    // that index we will mark it as invisible for this frame.
    for (let i = 0; i < smplMeshes.length; i++) {

        let found = false;
        frameData.forEach(person => {
            if (person.id == i) {
                found = true;
            }
        })

        if (!found) {
            // now fetch mesh from group
            const mesh_group = smplMeshes[i];
            const mesh = mesh_group.children[0];
            if (mesh.userData.visibility.invisibleIdx === undefined) {
                console.log("Person " + i + " track ends on frame " + timeIndex);

                mesh.userData.visibility.invisibleIdx = timeIndex;

                // Note this is duplicating the visible keyframes but that seems to blend without issue
                const opacityKF = createInvisibilityKeyframeTrack(mesh.name, mesh.userData.visibility.visibleIdx, mesh.userData.visibility.invisibleIdx);
                keyframeTracks.push(opacityKF);
            }
        }
    }

    console.log("smpMeshes length: " + smplMeshes.length + " frameData length: " + frameData.length);

    frameData.forEach(person => {

        const personId = person.id;
        const name = "person_" + personId;

        const vertices = person.verts;

        if (personId >= smplMeshes.length) {

            console.log("Creating new mesh for person " + personId);

            // This person isn't yet in the scene so we need to create a new mesh for them.
            const bufferGeometry = new THREE.BufferGeometry();

            // generate a unique color based on the personId. use a color map so that each person has a unique color
            function idToColor(personId) {
                const goldenRatioConjugate = 0.618033988749895;
                const hue = ((personId * goldenRatioConjugate) % 1) * 360;

                return new THREE.Color(`hsl(${hue}, 100%, 50%)`);
            }

            const mat = new THREE.MeshPhongMaterial({ color: idToColor(personId) });

            const positions = new Float32Array(vertices.length * 3);
            // Convert the coordinate system.
            vertices.forEach(function (vertice, i) {
                positions[i * 3] = vertice[0] / 1000.0;
                positions[i * 3 + 1] = vertice[1] / 1000.0;
                positions[i * 3 + 2] = vertice[2] / 1000.0;
            });
            const indices = new Uint16Array(faces.flat());
            bufferGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            bufferGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
            bufferGeometry.computeVertexNormals();

            // Prepare to add the morph targets and fill in the prior frames with invisible meshes
            bufferGeometry.morphTargetsRelative = false;
            bufferGeometry.morphAttributes.position = [];

            const mesh = new THREE.Mesh(bufferGeometry, mat);
            mesh.castShadow = false;
            mesh.baseMaterial = mesh.material;
            mesh.layers.enable(1);
            mesh.name = name;
            mesh.updateMorphTargets();

            const mesh_group = new THREE.Group()
            mesh_group.add(mesh);

            const opacityKF = createVisibilityKeyframeTrack(name, timeIndex);
            keyframeTracks.push(opacityKF);

            mesh.userData.visibility = { 'opacityTrack': opacityKF, 'visibleIdx': timeIndex, 'invisibleIdx': undefined };

            smplMeshes.push(mesh_group);

            scene.add(mesh_group);
        }

        // now fetch mesh from group
        const mesh_group = smplMeshes[personId];
        const mesh = mesh_group.children[0];
        const bufferGeometry = mesh.geometry;
        const morphVertices = new Float32Array(vertices.flat());

        const keyframeNumber = bufferGeometry.morphAttributes.position.length;

        // scale morphVertices down by dividing by 1000
        for (let i = 0; i < morphVertices.length; i++) {
            morphVertices[i] = morphVertices[i] / 1000.0;
        }
        bufferGeometry.morphAttributes.position.push(new THREE.Float32BufferAttribute(morphVertices, 3));

        // Let Three.js know that the existing morph attributes have been updated
        bufferGeometry.attributes.position.needsUpdate = true;

        function createVectorKeyframeTrack(timeIndex) {
            const dt = 5 / 30;
            const times = [timeIndex - 1, timeIndex, timeIndex + 1].map(t => t * dt);
            const values = [0, 1, 0];

            const track = new THREE.VectorKeyframeTrack(
                `scene/${name}.morphTargetInfluences[keyframe${keyframeNumber}]`,
                times,
                values,
                THREE.InterpolateLinear
            );

            return track;
        }

        // Create a new VectorKeyframeTrack and append it to the vectorKeyframeTracks list
        const track = createVectorKeyframeTrack(timeIndex);
        keyframeTracks.push(track);

        // Update the mesh.morphTargetDictionary with the keyframe number
        mesh.morphTargetDictionary[`keyframe${keyframeNumber}`] = keyframeNumber;

    });

    const trajectory = new THREE.AnimationClip('Action', -1, keyframeTracks);

    return { 'meshes': smplMeshes, 'tracks': keyframeTracks, 'trajectory': trajectory };
};

function createBiomechanicalMesh(meshData) {
    const boneMeshes = new Map();

    for (const [name, data] of Object.entries(meshData)) {
        console.log("Creating mesh for " + name);

        const vertices = data.vertices;
        const faces = data.faces;

        const geometry = new THREE.BufferGeometry();

        const positions = new Float32Array(vertices.length * 3);
        // Convert the coordinate system.
        vertices.forEach(function (vertice, i) {
            positions[i * 3] = vertice[0];
            positions[i * 3 + 1] = vertice[1];
            positions[i * 3 + 2] = vertice[2];
        });

        const indices = new Uint16Array(faces.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();

        const material = new THREE.MeshPhongMaterial({ color: 0x775533 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = name;
        boneMeshes.set(name, mesh);
    }

    return boneMeshes;
}

function createBiomechanicalTrajectory(trajectoryData, dt) {
    const tracks = [];

    for (const [name, data] of Object.entries(trajectoryData)) {
        console.log("Creating trajectory for " + name);

        const times = data.positions.map((_, index) => index * dt);
        const posTrack = new THREE.VectorKeyframeTrack(
            `scene/${name}.position`,
            times,
            data.positions.flat()
        );
        const rotTrack = new THREE.QuaternionKeyframeTrack(
            `scene/${name}.quaternion`,
            times,
            data.rotations.flat()
        );

        tracks.push(posTrack);
        tracks.push(rotTrack);
    }

    return new THREE.AnimationClip('Action', -1, tracks);
}

export {
    createScene, createTrajectory, createKeypointTrajectory, appendSmplFrame,
    createBiomechanicalMesh, createBiomechanicalTrajectory
};