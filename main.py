"""Main entry point for Qwen2.5-VL / Qwen3-VL application."""

import argparse
import sys

from qwen_vl.config import get_config, load_config
from qwen_vl.utils.logger import get_logger, setup_logging


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL / Qwen3-VL Production Service")
    parser.add_argument(
        "--check-hardware",
        action="store_true",
        help="Check hardware and exit",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check configuration and exit",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Start without Gradio UI (for API-only mode)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(
        level=config.logging.level,
        format_type=config.logging.format,
        file_path=config.logging.file_path,
    )

    logger = get_logger(__name__)

    if args.check_hardware:
        from qwen_vl.core.hardware_detection import HardwareDetector

        detector = HardwareDetector()
        detector.print_summary()
        return 0

    if args.check_config:
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"  Model: {config.model.model_id}")
        print(f"  Size: {config.model.size}")
        print(f"  Variant: {config.model.variant}")
        print(f"  Quantization: {config.model.quantization}")
        print(f"  Est. VRAM: {config.model.estimated_vram_gb} GB")
        print(f"  Max tokens: {config.inference.max_new_tokens}")
        print(f"  Server: {config.server.host}:{config.server.port}")
        print(f"  Log level: {config.logging.level}")
        print("=" * 60)
        return 0

    # Start the application
    logger.info("Starting Qwen VL service")
    logger.info(f"Model: {config.model.model_id}")

    if args.no_ui:
        print("=" * 60)
        print("Qwen VL Service - API Mode")
        print("=" * 60)
        print("Running without UI. Use --check-hardware or --check-config for diagnostics.")
        print("To start Gradio UI, run without --no-ui flag.")
        return 0

    # Launch Gradio UI
    print("=" * 60)
    print("Qwen VL Service - Starting Gradio UI")
    print("=" * 60)
    print(f"Model: {config.model.model_id}")
    print(f"Server: http://{config.server.host}:{config.server.port}")
    print("=" * 60)

    from qwen_vl.ui.gradio_app import launch_app

    launch_app(
        server_name=config.server.host,
        server_port=config.server.port,
        share=config.server.share,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

