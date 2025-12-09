import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { JOINTS } from "./mediaPipeHelper";
import { MarchingCubes } from "three/examples/jsm/objects/MarchingCubes.js";
import { getJoint } from "./getBody";
import {
  ONE_PLAYER_X_POSITIONS,
  TWO_PLAYER_X_POSITIONS,
  BODY_SCALE,
} from "./constants";

const bodyColors: THREE.Color[] = [
  new THREE.Color().setHex(0xff8ff4), // cyan
  new THREE.Color().setHex(0xfc5bef), // light pink
  new THREE.Color().setHex(0x0000ff), // blue
  new THREE.Color().setHex(0xff0000), // red
];

export const addCamera = (): THREE.PerspectiveCamera => {
  return new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    1,
    500
  );
};

export const addOrbitControls = (
  camera: THREE.PerspectiveCamera,
  canvas: HTMLElement
) => {
  return new OrbitControls(camera, canvas);
};

export const addDirectionalLight = (): THREE.DirectionalLight => {
  const white = 0xffffff;
  const directionalLight = new THREE.DirectionalLight(white, 3);
  directionalLight.position.set(100, 0, 100);
  directionalLight.target.position.set(5, 5, 0);
  return directionalLight;
};

function addBallWithPositionAndSize(
  xPos: number,
  yPos: number,
  strength: number,
  bodyIndex: number,
  numPlayers: number,
  skeletonMetaballs: MarchingCubes
) {
  // X positions for each dancer
  // Person 1, Person 2, AI 1, AI 2
  let xPositions = TWO_PLAYER_X_POSITIONS;

  // For 1 person (stay in one place):
  // have the extra 4th person in case I forget to change
  if (numPlayers === 1) {
    xPositions = ONE_PLAYER_X_POSITIONS;
  }

  // Use FULL 0-1 range, no padding at all
  let newXPos = 1 - xPos + xPositions[bodyIndex];

  skeletonMetaballs.addBall(
    newXPos * BODY_SCALE,
    (1 - yPos) * BODY_SCALE, // Subtracts pos from 1 to flip orientation
    0,
    strength,
    6,
    bodyColors[bodyIndex % bodyColors.length]
  );
}

// Adds balls on the line between two joints to create a continuous tube-like object
function addBallsBetweenJoints(
  joint1: { x: number; y: number },
  joint2: { x: number; y: number },
  numBalls: number,
  strength: number,
  bodyIndex: number,
  numPlayers: number,
  skeletonMetaballs: MarchingCubes
) {
  for (let i = 1; i <= numBalls; i++) {
    addBallWithPositionAndSize(
      joint2.x + (joint1.x - joint2.x) * (i / (numBalls + 1)),
      joint2.y + (joint1.y - joint2.y) * (i / (numBalls + 1)),
      strength,
      bodyIndex,
      numPlayers,
      skeletonMetaballs
    );
  }
}

// Averages the x, y position of two joints and returns a new joint
function averageJoints(
  joint1: { x: number; y: number },
  joint2: { x: number; y: number }
): { x: number; y: number } {
  return { x: (joint1.x + joint2.x) / 2, y: (joint1.y + joint2.y) / 2 };
}

// Create and return skeleton metaballs
export function createSkeletonMetaballs(RAPIER: any, world: any) {
  // Initialize bodies for joints
  const numSkeletonBodies = 23;
  const skeletonBodies: {
    color: THREE.Color;
    mesh:
      | THREE.Mesh<
          THREE.IcosahedronGeometry,
          THREE.MeshBasicMaterial,
          THREE.Object3DEventMap
        >
      | undefined;
    rigid: any;
    update?: () => THREE.Vector3;
    name?: string;
  }[] = [];
  for (let i = 0; i < numSkeletonBodies; i++) {
    const body = getJoint({ debug: true, RAPIER, world, xPos: 0, yPos: 0 });
    skeletonBodies.push(body);
  }

  //const normalMat = new THREE.MeshNormalMaterial();
  const matcapMat = new THREE.MeshMatcapMaterial({
    vertexColors: true,
  });
  //matcapMat.color = new THREE.Color().setHex(0x4deeea);
  const skeletonMetaballs = new MarchingCubes(
    96, // resolution of metaball,
    matcapMat,
    true, // enableUVs
    true, // enableColors
    90000 // max poly count
  );

  // Expand the bounding box to cover more space
  skeletonMetaballs.scale.setScalar(1);

  skeletonMetaballs.isolation = 800; // blobbiness or size. smaller number = bigger

  const legMultiplier = 1.5; // How many more balls between joints for the legs vs. the arms?

  skeletonMetaballs.userData = {
    // landmarks = currentPoses
    update(
      landmarks: any,
      strength: number,
      numBallsBetweenJoints: number,
      numPlayers: number
    ) {
      skeletonMetaballs.reset();

      // Only render AI bodies if they exist
      /*if (landmarks.length > 2) {
        landmarks = landmarks.slice(-2); // preserve last 2 elements
      }*/

      // loop through all existing rigid bodies, get add a metaball to each
      for (let j = 0; j < landmarks.length; j++) {
        skeletonBodies.forEach((_, i) => {
          // Skip all head landmarks, foot index, and hands
          if (
            i !== JOINTS.LEFT_PINKY &&
            i !== JOINTS.RIGHT_PINKY &&
            i !== JOINTS.LEFT_THUMB &&
            i !== JOINTS.RIGHT_THUMB
          ) {
            addBallWithPositionAndSize(
              landmarks[j][i].x,
              landmarks[j][i].y,
              strength,
              j,
              numPlayers,
              skeletonMetaballs
            );
          }
        });

        // Add skeleton head
        addBallWithPositionAndSize(
          landmarks[j][JOINTS.NOSE].x,
          landmarks[j][JOINTS.NOSE].y,
          8 * strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Add the skeleton's torso
        // Calculate X, Y average between left and right shoulder (x), left shoulder and left hip (y)

        // Torso
        addBallsBetweenJoints(
          averageJoints(
            landmarks[j][JOINTS.LEFT_SHOULDER],
            landmarks[j][JOINTS.RIGHT_SHOULDER]
          ),
          averageJoints(
            landmarks[j][JOINTS.LEFT_HIP],
            landmarks[j][JOINTS.RIGHT_HIP]
          ),
          numBallsBetweenJoints,
          5 * strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Right bicep
        addBallsBetweenJoints(
          landmarks[j][JOINTS.RIGHT_SHOULDER],
          landmarks[j][JOINTS.RIGHT_ELBOW],
          numBallsBetweenJoints,
          strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Left bicep
        addBallsBetweenJoints(
          landmarks[j][JOINTS.LEFT_SHOULDER],
          landmarks[j][JOINTS.LEFT_ELBOW],
          numBallsBetweenJoints,
          strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Right forearm
        addBallsBetweenJoints(
          landmarks[j][JOINTS.RIGHT_ELBOW],
          landmarks[j][JOINTS.RIGHT_WRIST],
          numBallsBetweenJoints,
          strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Left forearm
        addBallsBetweenJoints(
          landmarks[j][JOINTS.LEFT_ELBOW],
          landmarks[j][JOINTS.LEFT_WRIST],
          numBallsBetweenJoints,
          strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Right leg top 1 24, 26
        addBallsBetweenJoints(
          landmarks[j][JOINTS.RIGHT_HIP],
          landmarks[j][JOINTS.RIGHT_KNEE],
          Math.floor(numBallsBetweenJoints * legMultiplier),
          strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Right leg bottom 1 26, 28
        addBallsBetweenJoints(
          landmarks[j][JOINTS.RIGHT_KNEE],
          landmarks[j][JOINTS.RIGHT_ANKLE],
          Math.floor(numBallsBetweenJoints * legMultiplier),
          strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Left leg top 1 23, 25
        addBallsBetweenJoints(
          landmarks[j][JOINTS.LEFT_HIP],
          landmarks[j][JOINTS.LEFT_KNEE],
          Math.floor(numBallsBetweenJoints * legMultiplier),
          strength,
          j,
          numPlayers,
          skeletonMetaballs
        );

        // Left leg bottom 1 25, 27
        addBallsBetweenJoints(
          landmarks[j][JOINTS.LEFT_KNEE],
          landmarks[j][JOINTS.LEFT_ANKLE],
          Math.floor(numBallsBetweenJoints * legMultiplier),
          strength,
          j,
          numPlayers,
          skeletonMetaballs
        );
      }

      skeletonMetaballs.update();
    },
  };
  return skeletonMetaballs;
}
