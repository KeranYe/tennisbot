#!/usr/bin/env python3
import math
import sys
import time
from typing import Optional, Sequence
from evdev import InputDevice, ecodes

# try:
#     from evdev import InputDevice, ecodes
# except ImportError:
#     print(
#         "Missing dependency: python-evdev.\n"
#         "Install with: pip3 install evdev",
#         file=sys.stderr,
#     )
#     sys.exit(1)


def print_event_info(event) -> None:
    if event.type == ecodes.EV_KEY and event.value in (0, 1):
        state = "PRESSED" if event.value else "RELEASED"
        print(
            f"Button [{event.code:>3}] : {state:<8} "
            f"(code: 0x{event.code:x})"
        )
    elif event.type == ecodes.EV_ABS:
        normalized = event.value / 32767.0
        print(
            f"Axis   [{event.code:>3}] : (code: 0x{event.code:x})"
            f"{event.value:>6} "
            f"(normalized: {normalized:+.3f})"
        )


# def print_device_info(dev: InputDevice) -> None:
#     print("\nDevice Capabilities:")
#     print("==================")
#     print(f"Name: {dev.name}")
#     print(f"Physical Location: {dev.phys}")

#     caps_verbose = dev.capabilities(verbose=True, absinfo=True)
#     for ev_type, entries in caps_verbose.items():
#         type_code, type_name = ev_type
#         print(f"Event Type {type_code} ({type_name})")

#         for entry in entries:
#             if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], tuple):
#                 code_meta, absinfo = entry
#                 code, code_name = code_meta
#                 print(f"  Code {code} ({code_name})")
#                 if absinfo is not None:
#                     print(
#                         f"    Min: {absinfo.min} Max: {absinfo.max} "
#                         f"Fuzz: {absinfo.fuzz} Flat: {absinfo.flat}"
#                     )
#             elif isinstance(entry, tuple) and len(entry) == 2:
#                 code, code_name = entry
#                 print(f"  Code {code} ({code_name})")

#     print("==================\n")

# def print_device_info(dev: InputDevice) -> None:
#     print("\nDevice Capabilities:")
#     print("==================")
#     print(f"Name: {dev.name}")
#     print(f"Physical Location: {dev.phys}")

#     caps_verbose = dev.capabilities(verbose=True, absinfo=True)
#     for ev_type, entries in caps_verbose.items():
#         type_code, type_name = ev_type
#         print(f"Event Type {type_code} ({type_name})")

#         for entry in entries:
#             if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], tuple):
#                 code_meta, absinfo = entry
#                 # Handle variable-length tuples
#                 if len(code_meta) >= 2:
#                     code = code_meta[0]
#                     code_name = code_meta[1]
#                     print(f"  Code {code} ({code_name})")
#                     if absinfo is not None:
#                         print(
#                             f"    Min: {absinfo.min} Max: {absinfo.max} "
#                             f"Fuzz: {absinfo.fuzz} Flat: {absinfo.flat}"
#                         )
#             elif isinstance(entry, tuple) and len(entry) >= 2:
#                 code = entry[0]
#                 code_name = entry[1]
#                 print(f"  Code {code} ({code_name})")

#     print("==================\n")

def print_device_info(dev: InputDevice) -> None:
    print("\nDevice Capabilities:")
    print("==================")
    print(f"Name: {dev.name}")
    print(f"Physical Location: {dev.phys}")

    caps_verbose = dev.capabilities(verbose=True, absinfo=True)
    for ev_type, entries in caps_verbose.items():
        type_code, type_name = ev_type
        print(f"Event Type {type_code} ({type_name})")

        for entry in entries:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], tuple):
                code_meta, absinfo = entry
                # Handle variable-length tuples
                if len(code_meta) >= 2:
                    code = code_meta[0]
                    code_name = code_meta[1]
                    print(f"  Code {code} ({code_name})")
                    # Check if absinfo is an AbsInfo object (has 'min' attribute)
                    if absinfo is not None and hasattr(absinfo, 'min'):
                        print(
                            f"    Min: {absinfo.min} Max: {absinfo.max} "
                            f"Fuzz: {absinfo.fuzz} Flat: {absinfo.flat}"
                        )
            elif isinstance(entry, tuple) and len(entry) >= 2:
                code = entry[0]
                code_name = entry[1]
                print(f"  Code {code} ({code_name})")

    print("==================\n")


def find_matching_device(device_name: str, candidates: Sequence[str]) -> Optional[InputDevice]:
    for path in candidates:
        try:
            dev = InputDevice(path)
        except OSError:
            continue

        if dev.name and device_name in dev.name:
            print(f"Connected to gamepad: {dev.name}")
            print(f"Using device: {path}")
            return dev

        dev.close()

    return None


def main() -> int:
    device_name = "8Bitdo SF30 Pro"  # 8Bitdo SF30 Pro, Logitech Gamepad F710

    candidates = [
        "/dev/input/event0",
        "/dev/input/event1",
        "/dev/input/event2",
        "/dev/input/event3",
        "/dev/input/event4",
        "/dev/input/event28",
        "/dev/input/event29",
        "/dev/input/js0",
        "/dev/input/js1",
    ]

    dev = find_matching_device(device_name, candidates)
    if dev is None:
        print(
            f"No gamepad {device_name} found.\n"
            "Please connect your gamepad and try again.",
            file=sys.stderr,
        )
        return 1

    print_device_info(dev)

    try:
        dev.grab()
    except OSError:
        print("Warning: Could not grab device exclusively", file=sys.stderr)

    print(
        "\n=== Gamepad Event Detector ===\n"
        f"Device: {device_name}\n"
        "Press buttons and move sticks to see their codes\n"
        "- Button events show press/release and hex codes\n"
        "- Axis events show raw and normalized values (-1.0 to 1.0)\n"
        "Press Ctrl+C to exit\n"
    )

    try:
        while True:
            event = dev.read_one()
            if event is not None:
                print_event_info(event)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            dev.ungrab()
        except OSError:
            pass
        dev.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
