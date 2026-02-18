"""
This script converts LAMMPS trajectory files (.lammpstrj) to video files (.avi)
using the OVITO python module.
"""

import argparse
import os
import sys
from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer, OpenGLRenderer
from ovito.modifiers import ComputePropertyModifier

def convert_video(input_file, output_file=None, renderer_type="opengl", scale_factor=0.8):
    """
    Convert a LAMMPS trajectory file to an AVI video.

    Args:
        input_file: Path to the .lammpstrj file
        output_file: Path to the output .avi file (optional)
        renderer_type: 'opengl' or 'tachyon'
        scale_factor: Scaling factor for particle radius
    """
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".avi"

    print(
        f"Converting '{input_file}' to '{output_file}' "
        f"using {renderer_type} with radius scale factor {scale_factor}..."
    )

    renderer = OpenGLRenderer() if renderer_type == "opengl" else TachyonRenderer()

    try:
        # Load the LAMMPS trajectory
        pipeline = import_file(input_file)

        # Check for existing 'Radius' property
        data = pipeline.compute()
        if data.particles is not None and "Radius" in data.particles.keys():
            # Apply scaling to existing Radius property
            pipeline.modifiers.append(
                ComputePropertyModifier(
                    output_property="Radius", expressions=[f"Radius * {scale_factor}"]
                )
            )
        else:
            # Radius property missing, likely using OVITO default (0.8).
            # We must explicitly set the new radius.
            pipeline.modifiers.append(
                ComputePropertyModifier(
                    output_property="Radius", expressions=[f"0.8 * {scale_factor}"]
                )
            )

        pipeline.add_to_scene()

        # Set up the viewport
        viewport = Viewport()
        viewport.type = Viewport.Type.Perspective
        viewport.zoom_all()

        # Render directly to video
        # Defaulting to 800x600 resolution and 30fps
        try:
            viewport.render_anim(
                filename=output_file, size=(800, 600), fps=30, renderer=renderer
            )
        except RuntimeError as e:  # Catching RuntimeError for Ovito rendering issues
            if renderer_type == "opengl":
                print(
                    f"OpenGL rendering failed ({e}). "
                    "Falling back to CPU (TachyonRenderer)..."
                )
                viewport.render_anim(
                    filename=output_file,
                    size=(800, 600),
                    fps=30,
                    renderer=TachyonRenderer(),
                )
            else:
                raise e

        pipeline.remove_from_scene()
        print("Conversion successful.")

    except (RuntimeError, ValueError, OSError) as e:
        print(f"Error converting file: {e}")


def main():
    """Main function to parse arguments and convert files."""
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS trajectory (.lammpstrj) to AVI video using OVITO."
    )
    parser.add_argument(
        "path", help="Path to a .lammpstrj file or a directory containing them."
    )
    parser.add_argument(
        "--renderer",
        choices=["opengl", "tachyon"],
        default="opengl",
        help="Renderer to use: 'opengl' (GPU) or 'tachyon' (CPU/Raytracing)."
             " Default is 'opengl'.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.8,
        help="Scale factor for particle radius. Default is 0.8.",
    )

    args = parser.parse_args()
    path = args.path
    renderer = args.renderer
    scale = args.scale

    if os.path.isfile(path):
        if path.endswith(".lammpstrj"):
            convert_video(path, renderer_type=renderer, scale_factor=scale)
        else:
            print("Error: File provided is not a .lammpstrj file.")

    elif os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".lammpstrj")]
        if not files:
            print("No .lammpstrj files found in the directory.")
            return

        print(f"Found {len(files)} files processing...")
        files.sort()

        for f in files:
            full_path = os.path.join(path, f)
            convert_video(full_path, renderer_type=renderer, scale_factor=scale)

        print(f"Error: '{path}' is not a valid file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
