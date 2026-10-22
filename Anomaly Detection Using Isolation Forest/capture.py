import threading
import time
from scapy.all import sniff, conf
from features import packet_callback

# =========================
# GLOBAL STATE
# =========================

packet_store = []
capturing = False
sniffer_thread = None


# =========================
# START CAPTURE
# =========================

def start_capture(interface=None):
    """
    Start packet capture in a separate thread
    """

    global capturing, sniffer_thread, packet_store

    if capturing:
        print("⚠️ Capture already running")
        return

    # Auto-detect interface if not provided
    if interface is None:
        interface = conf.iface

    packet_store.clear()
    capturing = True

    def sniff_packets():
        try:
            sniff(
                iface=interface,
                prn=lambda pkt: packet_callback(pkt, packet_store),
                store=False,
                stop_filter=lambda x: not capturing,
                timeout=60   # 🔥 prevents infinite blocking
            )
        except Exception as e:
            print(f"❌ Sniff error: {e}")

    sniffer_thread = threading.Thread(target=sniff_packets, daemon=True)
    sniffer_thread.start()

    print(f"🚀 Capture started on interface: {interface}")


# =========================
# STOP CAPTURE
# =========================

def stop_capture():
    """
    Stop packet capture safely
    """

    global capturing, sniffer_thread

    if not capturing:
        print("⚠️ Capture is not running")
        return []

    capturing = False
    print("🛑 Stopping capture...")

    # 🔥 Wait for thread to exit properly
    if sniffer_thread is not None:
        sniffer_thread.join(timeout=2)

    print(f"✅ Capture stopped. Total packets: {len(packet_store)}")

    return packet_store.copy()


# =========================
# LIVE STATS
# =========================

def get_packet_count():
    return len(packet_store)


def get_capture_stats():
    """
    Extendable stats for dashboard
    """
    return {
        "total_packets": len(packet_store),
    }


# =========================
# RESET
# =========================

def reset_capture():
    global packet_store
    packet_store.clear()