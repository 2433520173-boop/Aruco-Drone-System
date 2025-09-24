from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np

# --- CẤU HÌNH MỚI: Quét nhiều loại từ điển Aruco ---
ARUCO_DICTIONARIES = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
}
# -----------------------------------------------

# --- QUẢN LÝ TRẠNG THÁI TOÀN CỤC ---
app_mode = 'ghi_nho'
target_ids = set()
candidate_ids = set()
detected_markers_ordered = []
seen_ids = set()

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():
    """Hàm tạo luồng video và xử lý logic theo từng chế độ."""
    global candidate_ids, app_mode

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Tối ưu hóa để chạy mượt
        scale = 640 / frame.shape[1]
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # --- THAY ĐỔI LOGIC NHẬN DIỆN ĐỂ QUÉT NHIỀU LOẠI ARUCO ---
        all_corners = []
        all_ids = []
        
        # Lặp qua tất cả các từ điển Aruco được định nghĩa
        for (aruco_name, dictionary_id) in ARUCO_DICTIONARIES.items():
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            (corners_detected, ids_detected, rejected) = cv2.aruco.detectMarkers(frame, dictionary)
            
            if ids_detected is not None:
                all_ids.extend(ids_detected) # Gộp các ID tìm thấy
                all_corners.extend(corners_detected) # Gộp các góc tìm thấy
        
        # Chuẩn hóa lại tên biến để tương thích với code phía dưới
        ids = np.array(all_ids) if all_ids else None
        corners = all_corners
        # --- KẾT THÚC THAY ĐỔI ---
        
        if app_mode == 'ghi_nho':
            cv2.putText(frame, "Buoc 1: Dua cac ma can ghi nho vao khung hinh", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            candidate_ids.clear()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for marker_id_array in ids:
                    candidate_ids.add(marker_id_array[0])
                
                if candidate_ids:
                    cv2.putText(frame, f"San sang them {len(candidate_ids)} ma", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif app_mode == 'nhan_dien':
            cv2.putText(frame, "Buoc 2: Nhan dien... Drone dang hoat dong.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if ids is not None:
                for marker_id_array in ids:
                    marker_id = marker_id_array[0]
                    if marker_id not in seen_ids:
                        seen_ids.add(marker_id)
                        position = len(detected_markers_ordered) + 1
                        is_target = marker_id in target_ids
                        detected_markers_ordered.append({
                            'id': marker_id,
                            'position': position,
                            'is_target': is_target
                        })

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    marker_info = next((item for item in detected_markers_ordered if item["id"] == marker_id), None)
                    if marker_info:
                        cx = int(np.mean(corners[i][0, :, 0]))
                        cy = int(np.mean(corners[i][0, :, 1]))
                        pos_text = f"TT: {marker_info['position']}"
                        cv2.putText(frame, pos_text, (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        if marker_info['is_target']:
                             cv2.putText(frame, "--> DA TIM THAY!", (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Các route còn lại giữ nguyên
@app.route('/')
def index():
    sorted_results = sorted(detected_markers_ordered, key=lambda x: x['position'])
    return render_template('index.html', 
                           app_mode=app_mode, 
                           target_ids=sorted(list(target_ids)), 
                           results=sorted_results)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_targets', methods=['POST'])
def add_targets():
    if candidate_ids:
        target_ids.update(candidate_ids)
    return redirect(url_for('index'))

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global app_mode
    app_mode = 'nhan_dien'
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    global app_mode, target_ids, candidate_ids, detected_markers_ordered, seen_ids
    app_mode = 'ghi_nho'
    target_ids.clear()
    candidate_ids.clear()
    detected_markers_ordered.clear()
    seen_ids.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)