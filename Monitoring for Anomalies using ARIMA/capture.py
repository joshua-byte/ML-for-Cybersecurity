
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def list_interfaces() -> list[str]:
    """Return the network interfaces available to Scapy."""
    try:
        from scapy.all import get_if_list
    except ImportError as exc:
        raise RuntimeError(
            "Scapy is not installed. Install it with: pip install scapy"
        ) from exc

    return list(get_if_list())


def _validate_capture_settings(
    duration_seconds: int,
    interval_seconds: int,
) -> None:
    """Validate capture duration and aggregation interval."""
    if not isinstance(duration_seconds, int) or isinstance(duration_seconds, bool):
        raise ValueError("Capture duration must be an integer.")
    if not 10 <= duration_seconds <= 600:
        raise ValueError("Capture duration must be between 10 and 600 seconds.")

    if not isinstance(interval_seconds, int) or isinstance(interval_seconds, bool):
        raise ValueError("Aggregation interval must be an integer.")
    if interval_seconds != 60:
        raise ValueError(
            "The current ARIMA model is calibrated for 60-second intervals. "
            "Use interval_seconds=60 unless a compatible model is trained."
        )


def capture_packets(
    duration_seconds: int = 60,
    interface: str | None = None,
) -> list[tuple[float, int]]:
    """
    Capture packets for a fixed duration.

    Returns:
        A list of (Unix timestamp, packet length) tuples.

    Notes:
        - Capture runs on the machine running Python.
        - Packet contents are not stored.
        - Appropriate OS permissions may be required.
        - Only capture networks you are authorized to monitor.
    """
    _validate_capture_settings(duration_seconds, 60)

    try:
        from scapy.all import sniff, get_if_list
    except ImportError as exc:
        raise RuntimeError(
            "Scapy is not installed. Install it with: pip install scapy"
        ) from exc

    available_interfaces = list(get_if_list())

    if interface:
        if interface not in available_interfaces:
            raise ValueError(
                f"Interface '{interface}' was not found. Available interfaces: "
                + ", ".join(available_interfaces)
            )

    packets: list[tuple[float, int]] = []

    def packet_callback(packet: Any) -> None:
        timestamp = float(getattr(packet, "time", time.time()))
        packet_length = int(len(packet))
        packets.append((timestamp, packet_length))

    try:
        sniff(
            iface=interface or None,
            prn=packet_callback,
            store=False,
            timeout=duration_seconds,
        )
    except PermissionError as exc:
        raise PermissionError(
            "Packet capture permission was denied. Check interface selection "
            "and operating-system capture permissions."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Packet capture failed: {exc}") from exc

    return packets


def aggregate_packets(
    packets: list[tuple[float, int]],
    interval_seconds: int = 60,
) -> pd.DataFrame:
    """
    Aggregate captured packets into fixed-width intervals.

    The output contains:
        interval_start_utc
        packet_count
        total_bytes

    Empty intervals between the first and last captured packet are retained
    with zero counts. Intervals before the first packet and after the last
    packet are not inferred by this function.
    """
    _validate_capture_settings(60, interval_seconds)

    columns = ["interval_start_utc", "packet_count", "total_bytes"]

    if not packets:
        return pd.DataFrame(columns=columns)

    clean_packets: list[tuple[float, int]] = []

    for timestamp, packet_length in packets:
        if not isinstance(timestamp, (int, float)) or not pd.notna(timestamp):
            raise ValueError("Packet timestamps must be valid Unix timestamps.")
        if packet_length < 0:
            raise ValueError("Packet lengths cannot be negative.")

        clean_packets.append((float(timestamp), int(packet_length)))

    first_timestamp = min(timestamp for timestamp, _ in clean_packets)
    last_timestamp = max(timestamp for timestamp, _ in clean_packets)

    # Anchor bins to the first packet timestamp for this capture.
    bin_count = int((last_timestamp - first_timestamp) // interval_seconds) + 1

    packet_counts = [0] * bin_count
    byte_counts = [0] * bin_count

    for timestamp, packet_length in clean_packets:
        bin_index = int((timestamp - first_timestamp) // interval_seconds)
        bin_index = min(max(bin_index, 0), bin_count - 1)

        packet_counts[bin_index] += 1
        byte_counts[bin_index] += packet_length

    interval_starts = [
        datetime.fromtimestamp(
            first_timestamp + index * interval_seconds,
            tz=timezone.utc,
        ).isoformat()
        for index in range(bin_count)
    ]

    return pd.DataFrame(
        {
            "interval_start_utc": interval_starts,
            "packet_count": packet_counts,
            "total_bytes": byte_counts,
        }
    )


def capture_traffic(
    duration_seconds: int = 60,
    interface: str | None = None,
    interval_seconds: int = 60,
) -> pd.DataFrame:
    """
    Capture traffic and return aggregated data ready for app.py.

    Example:
        traffic = capture_traffic(
            duration_seconds=60,
            interface=None,
            interval_seconds=60,
        )
    """
    _validate_capture_settings(duration_seconds, interval_seconds)

    packets = capture_packets(
        duration_seconds=duration_seconds,
        interface=interface,
    )

    return aggregate_packets(
        packets,
        interval_seconds=interval_seconds,
    )
