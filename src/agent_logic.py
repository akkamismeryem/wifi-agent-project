from typing import Dict, Any


def calculate_risk_score(row: Dict[str, Any]) -> Dict[str, Any]:
    traffic_risk = 0
    error_risk = 0
    packet_risk = 0
    content_risk = 0
    host_risk = 0
    anomaly_risk = 0

    count = float(row.get("count", 0))
    srv_count = float(row.get("srv_count", 0))
    same_srv_rate = float(row.get("same_srv_rate", 0))
    diff_srv_rate = float(row.get("diff_srv_rate", 0))
    srv_diff_host_rate = float(row.get("srv_diff_host_rate", 0))

    serror_rate = float(row.get("serror_rate", 0))
    srv_serror_rate = float(row.get("srv_serror_rate", 0))
    rerror_rate = float(row.get("rerror_rate", 0))
    srv_rerror_rate = float(row.get("srv_rerror_rate", 0))

    src_bytes = float(row.get("src_bytes", 0))
    dst_bytes = float(row.get("dst_bytes", 0))
    duration = float(row.get("duration", 0))

    num_failed_logins = float(row.get("num_failed_logins", 0))
    logged_in = float(row.get("logged_in", 0))
    is_guest_login = float(row.get("is_guest_login", 0))
    num_compromised = float(row.get("num_compromised", 0))
    root_shell = float(row.get("root_shell", 0))
    su_attempted = float(row.get("su_attempted", 0))
    num_root = float(row.get("num_root", 0))
    num_file_creations = float(row.get("num_file_creations", 0))
    num_shells = float(row.get("num_shells", 0))
    num_access_files = float(row.get("num_access_files", 0))
    hot = float(row.get("hot", 0))

    dst_host_count = float(row.get("dst_host_count", 0))
    dst_host_srv_count = float(row.get("dst_host_srv_count", 0))
    dst_host_same_srv_rate = float(row.get("dst_host_same_srv_rate", 0))
    dst_host_diff_srv_rate = float(row.get("dst_host_diff_srv_rate", 0))
    dst_host_same_src_port_rate = float(row.get("dst_host_same_src_port_rate", 0))
    dst_host_srv_diff_host_rate = float(row.get("dst_host_srv_diff_host_rate", 0))
    dst_host_serror_rate = float(row.get("dst_host_serror_rate", 0))
    dst_host_srv_serror_rate = float(row.get("dst_host_srv_serror_rate", 0))
    dst_host_rerror_rate = float(row.get("dst_host_rerror_rate", 0))
    dst_host_srv_rerror_rate = float(row.get("dst_host_srv_rerror_rate", 0))

    land = float(row.get("land", 0))
    wrong_fragment = float(row.get("wrong_fragment", 0))
    urgent = float(row.get("urgent", 0))

    # ---------- TRAFFIC ----------
    if count > 100:
        traffic_risk += 4
    elif count > 50:
        traffic_risk += 3
    elif count > 20:
        traffic_risk += 2

    if srv_count > 100:
        traffic_risk += 3
    elif srv_count > 50:
        traffic_risk += 2

    if same_srv_rate > 0.9:
        traffic_risk += 2

    if diff_srv_rate > 0.7:
        traffic_risk += 2

    if srv_diff_host_rate > 0.5:
        traffic_risk += 2

    # ---------- ERROR ----------
    if serror_rate > 0.8:
        error_risk += 4
    elif serror_rate > 0.5:
        error_risk += 2

    if srv_serror_rate > 0.8:
        error_risk += 3

    if rerror_rate > 0.5:
        error_risk += 2

    if srv_rerror_rate > 0.5:
        error_risk += 2

    # ---------- PACKET ----------
    if src_bytes == 0 and dst_bytes == 0:
        packet_risk += 3
    elif src_bytes < 100 and dst_bytes < 100:
        packet_risk += 1

    if duration == 0:
        packet_risk += 2

    # ---------- CONTENT ----------
    if num_failed_logins > 5:
        content_risk += 4

    if logged_in == 0 and num_failed_logins > 0:
        content_risk += 2

    if is_guest_login == 1:
        content_risk += 1

    if num_compromised > 0:
        content_risk += 4

    if root_shell == 1:
        content_risk += 10

    if su_attempted == 1:
        content_risk += 5

    if num_root > 0:
        content_risk += 6

    if num_file_creations > 2:
        content_risk += 3

    if num_shells > 0:
        content_risk += 5

    if num_access_files > 2:
        content_risk += 2

    if hot > 0:
        content_risk += 2

    # ---------- HOST ----------
    if dst_host_count > 200:
        host_risk += 3

    if dst_host_srv_count > 200:
        host_risk += 3

    if dst_host_same_srv_rate > 0.9:
        host_risk += 2

    if dst_host_diff_srv_rate > 0.7:
        host_risk += 2

    if dst_host_same_src_port_rate > 0.9:
        host_risk += 1

    if dst_host_srv_diff_host_rate > 0.5:
        host_risk += 2

    if dst_host_serror_rate > 0.8:
        host_risk += 3

    if dst_host_srv_serror_rate > 0.8:
        host_risk += 3

    if dst_host_rerror_rate > 0.5:
        host_risk += 2

    if dst_host_srv_rerror_rate > 0.5:
        host_risk += 2

    # ---------- ANOMALY ----------
    if land == 1:
        anomaly_risk += 5

    if wrong_fragment > 0:
        anomaly_risk += 3

    if urgent > 0:
        anomaly_risk += 2

    total_risk = (
        traffic_risk +
        error_risk +
        packet_risk +
        content_risk +
        host_risk +
        anomaly_risk
    )

    if total_risk < 5:
        risk_level = "NORMAL"
    elif total_risk < 10:
        risk_level = "LOW RISK"
    elif total_risk < 20:
        risk_level = "MEDIUM RISK"
    elif total_risk < 30:
        risk_level = "HIGH RISK"
    else:
        risk_level = "CRITICAL RISK"

    # ---------- ATTACK HINT ----------
    # U2R
    if (
        root_shell == 1
        or su_attempted == 1
        or num_root > 0
        or num_shells > 0
        or content_risk >= 10
    ):
        attack_hint = "U2R / yetki yükseltme veya sistem ele geçirme benzeri davranış"

    # R2L
    elif (
        num_failed_logins > 5
        or (logged_in == 0 and num_failed_logins > 2)
        or (is_guest_login == 1 and num_compromised > 0)
    ):
        attack_hint = "R2L / yetkisiz erişim veya brute-force benzeri davranış"

    # DoS
    elif (
        (traffic_risk >= 4 and error_risk >= 4)
        or (count > 50 and same_srv_rate > 0.9 and (src_bytes == 0 or dst_bytes == 0))
        or (dst_host_same_srv_rate > 0.9 and dst_host_serror_rate > 0.8)
    ):
        attack_hint = "DoS / flood benzeri davranış"

    # PROBE
    elif (
        (
            diff_srv_rate > 0.7
            or srv_diff_host_rate > 0.5
            or dst_host_diff_srv_rate > 0.7
            or dst_host_srv_diff_host_rate > 0.5
        )
        and count >= 5
        and error_risk <= 4
        and content_risk == 0
        and not (same_srv_rate > 0.9 and count > 50)
    ):
        attack_hint = "PROBE / tarama veya keşif benzeri davranış"

    # Genel şüpheli
    elif total_risk >= 10:
        attack_hint = "Şüpheli trafik / manuel inceleme önerilir"

    else:
        attack_hint = "Normale yakın trafik"

    return {
        "traffic_risk": traffic_risk,
        "error_risk": error_risk,
        "packet_risk": packet_risk,
        "content_risk": content_risk,
        "host_risk": host_risk,
        "anomaly_risk": anomaly_risk,
        "total_risk": total_risk,
        "risk_level": risk_level,
        "attack_hint": attack_hint,
    }