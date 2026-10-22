import time
from scapy.all import IP, TCP, UDP

# =========================
# PACKET → DICT CONVERSION
# =========================

def process_packet(packet):
    """
    Convert a Scapy packet into a structured dictionary
    """

    pkt_dict = {}

    # Timestamp
    pkt_dict["timestamp"] = time.time()

    # =========================
    # IP LAYER
    # =========================
    if not packet.haslayer(IP):
        return None

    pkt_dict["src_ip"] = packet[IP].src
    pkt_dict["dst_ip"] = packet[IP].dst

    # Packet length
    pkt_dict["length"] = len(packet)

    # Defaults
    pkt_dict["src_port"] = 0
    pkt_dict["dst_port"] = 0
    pkt_dict["flags"] = ""
    pkt_dict["protocol"] = "OTHER"

    # =========================
    # TCP
    # =========================
    if packet.haslayer(TCP):
        pkt_dict["protocol"] = "TCP"
        pkt_dict["src_port"] = packet[TCP].sport
        pkt_dict["dst_port"] = packet[TCP].dport

        # 🔥 PROPER FLAG EXTRACTION (IMPORTANT)
        flags = packet[TCP].flags

        flag_str = ""
        if flags & 0x02: flag_str += "S"   # SYN
        if flags & 0x10: flag_str += "A"   # ACK
        if flags & 0x01: flag_str += "F"   # FIN
        if flags & 0x04: flag_str += "R"   # RST
        if flags & 0x08: flag_str += "P"   # PSH

        pkt_dict["flags"] = flag_str

    # =========================
    # UDP
    # =========================
    elif packet.haslayer(UDP):
        pkt_dict["protocol"] = "UDP"
        pkt_dict["src_port"] = packet[UDP].sport
        pkt_dict["dst_port"] = packet[UDP].dport

    # =========================
    # OPTIONAL: DIRECTION (useful later)
    # =========================
    if pkt_dict["src_ip"].startswith("192.") or pkt_dict["src_ip"].startswith("10."):
        pkt_dict["direction"] = "outgoing"
    else:
        pkt_dict["direction"] = "incoming"

    return pkt_dict


# =========================
# BULK PACKET PROCESSING
# =========================

def extract_packet_list(scapy_packets):
    """
    Convert list of scapy packets → list of dictionaries
    """

    processed_packets = []

    for pkt in scapy_packets:
        try:
            p = process_packet(pkt)
            if p:
                processed_packets.append(p)
        except Exception:
            continue

    return processed_packets


# =========================
# LIVE CAPTURE CALLBACK
# =========================

def packet_callback(packet, packet_store):
    """
    Used in sniff(prn=...) for live capture
    """

    try:
        pkt = process_packet(packet)
        if pkt:
            packet_store.append(pkt)
    except Exception:
        pass


# =========================
# DEBUG
# =========================

def print_sample(packets, n=5):
    print("\n=== SAMPLE PACKETS ===")
    for pkt in packets[:n]:
        print(pkt)