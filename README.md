# Unity Barracuda Inference PoseNet
This Unity package extends the functionality of the [barracuda-inference-base](https://github.com/cj-mills/unity-barracuda-inference-base) package to perform 2D human pose estimation using PoseNet models. 

## Demo Video
https://user-images.githubusercontent.com/9126128/232156999-217ed86b-6496-4f31-8c54-5fc60e68e3cd.mp4

## Features

- Easy integration with PoseNet models
- Utilizes Unity's Barracuda engine for efficient inference
- Supports various worker types
- Processes output to generate pose skeletons

## Getting Started

### Prerequisites

- Unity game engine

### Installation

You can install the Barracuda Inference PoseNet package using the Unity Package Manager:

1. Open your Unity project.
2. Go to Window > Package Manager.
3. Click the "+" button in the top left corner, and choose "Add package from git URL..."
4. Enter the GitHub repository URL: `https://github.com/cj-mills/unity-barracuda-inference-posenet.git`
5. Click "Add". The package will be added to your project.

For Unity versions older than 2021.1, add the Git URL to the `manifest.json` file in your project's `Packages` folder as a dependency:

```json
{
  "dependencies": {
    "com.cj-mills.barracuda-inference-posenet": "https://github.com/cj-mills/unity-barracuda-inference-posenet.git",
    // other dependencies...
  }
}
```



## License

This project is licensed under the MIT License. See the [LICENSE](Documentation~/LICENSE) file for details.
