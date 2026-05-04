#!/usr/bin/env python3
"""
Python скрипт для запуска MultiKernelBench API сервиса.
"""

import os
import sys
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start MultiKernelBench API server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "1")),
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Проверка наличия app/main.py
    if not os.path.exists("app/main.py"):
        print("Error: app/main.py not found")
        print("Please run this script from the bench_api directory")
        sys.exit(1)

    # Проверка наличия MultiKernelBench
    if not os.path.exists("../MultiKernelBench"):
        print("Warning: MultiKernelBench not found")
        print("The API may not work correctly without MultiKernelBench")

    print("=" * 60)
    print("MultiKernelBench API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)
    print("Starting server...")
    print("")

    # Запуск сервера
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
