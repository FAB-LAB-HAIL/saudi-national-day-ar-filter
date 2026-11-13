import cv2
import mediapipe as mp
import math
import numpy as np
import time

def overlay_transparent(bg, overlay, x, y):
    h, w = overlay.shape[0], overlay.shape[1]

    if x >= bg.shape[1] or y >= bg.shape[0] or x + w <= 0 or y + h <= 0:
        return bg

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg.shape[1], x + w), min(bg.shape[0], y + h)

    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    alpha_s = overlay[oy1:oy2, ox1:ox2, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        bg[y1:y2, x1:x2, c] = (alpha_s * overlay[oy1:oy2, ox1:ox2, c] +
                               alpha_l * bg[y1:y2, x1:x2, c])
    return bg


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)

# تحميل الصور بشكل مستقل
img_glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)
img_hat = cv2.imread('hat.png', cv2.IMREAD_UNCHANGED)
sticker_img = cv2.imread('sticker.png', cv2.IMREAD_UNCHANGED)
frame_img = cv2.imread('frame.png', cv2.IMREAD_UNCHANGED)

# الحالات الأربعة:
# 1: نظارة فقط
# 2: كاب + ستيكرات
# 3: ستيكرات فقط
# 4: كاب فقط
filter_keys = ['1', '2', '3', '4']
current_filter_index = 0
current_filter_key = filter_keys[current_filter_index]
last_switch_time = time.time()

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("National Day AR Filter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("National Day AR Filter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h_img, w_img, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # تحديد النقاط الأساسية
            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            forehead = face_landmarks.landmark[10]
            left_ear = face_landmarks.landmark[454]
            right_ear = face_landmarks.landmark[234]

            nx, ny = int(nose.x * w_img), int(nose.y * h_img)
            lx, ly = int(left_eye.x * w_img), int(left_eye.y * h_img)
            rx, ry = int(right_eye.x * w_img), int(right_eye.y * h_img)
            fx, fy = int(forehead.x * w_img), int(forehead.y * h_img)
            lex, ley = int(left_ear.x * w_img), int(left_ear.y * h_img)
            rex, rey = int(right_ear.x * w_img), int(right_ear.y * h_img)

            eye_dist = math.hypot(rx - lx, ry - ly)
            angle = math.degrees(math.atan2(ry - ly, rx - lx))
            is_tilting = abs(angle) > 15

            # تبديل الفلاتر بإمالة الرأس
            if is_tilting and (time.time() - last_switch_time > 2.0):
                current_filter_index = (current_filter_index + 1) % len(filter_keys)
                current_filter_key = filter_keys[current_filter_index]
                last_switch_time = time.time()

            # تحديد ماذا يتم رسمه بناءً على الحالة الحالية
            draw_glasses = (current_filter_key == '1')
            draw_hat = (current_filter_key in ['2', '4'])
            draw_stickers = (current_filter_key in ['2', '3'])

            # ==========================================
            # 1. رسم الستيكرات (بإعداداتك الدقيقة)
            # ==========================================
            if draw_stickers and sticker_img is not None:
                orig_s_h, orig_s_w = sticker_img.shape[:2]

                placement_ratio = 0.8
                scale_multiplier = 0.60
                y_offset = 50

                # الخد الأيمن
                rx_center = int(nx + (rex - nx) * placement_ratio)
                ry_center = int(ny + (rey - ny) * placement_ratio) + y_offset
                right_side_dist = math.hypot(rex - nx, rey - ny)
                sr_scale_factor = (right_side_dist / orig_s_w) * scale_multiplier
                new_sr_w, new_sr_h = int(orig_s_w * sr_scale_factor), int(orig_s_h * sr_scale_factor)

                if new_sr_w > 0 and new_sr_h > 0:
                    resized_sr = cv2.resize(sticker_img, (new_sr_w, new_sr_h))
                    M_sr = cv2.getRotationMatrix2D((new_sr_w // 2, new_sr_h // 2), -angle, 1.0)
                    rotated_sr = cv2.warpAffine(resized_sr, M_sr, (new_sr_w, new_sr_h),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=(0, 0, 0, 0))
                    offset_x_right = rx_center - int(new_sr_w / 2)
                    offset_y_right = ry_center - int(new_sr_h / 2)
                    img = overlay_transparent(img, rotated_sr, offset_x_right, offset_y_right)

                # الخد الأيسر
                lx_center = int(nx + (lex - nx) * placement_ratio)
                ly_center = int(ny + (ley - ny) * placement_ratio) + y_offset
                left_side_dist = math.hypot(lex - nx, ley - ny)
                sl_scale_factor = (left_side_dist / orig_s_w) * scale_multiplier
                new_sl_w, new_sl_h = int(orig_s_w * sl_scale_factor), int(orig_s_h * sl_scale_factor)

                if new_sl_w > 0 and new_sl_h > 0:
                    resized_sl = cv2.resize(sticker_img, (new_sl_w, new_sl_h))
                    M_sl = cv2.getRotationMatrix2D((new_sl_w // 2, new_sl_h // 2), -angle, 1.0)
                    rotated_sl = cv2.warpAffine(resized_sl, M_sl, (new_sl_w, new_sl_h),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=(0, 0, 0, 0))
                    offset_x_left = lx_center - int(new_sl_w / 2)
                    offset_y_left = ly_center - int(new_sl_h / 2)
                    img = overlay_transparent(img, rotated_sl, offset_x_left, offset_y_left)

            # ==========================================
            # 2. رسم النظارة
            # ==========================================
            if draw_glasses and img_glasses is not None:
                orig_h, orig_w = img_glasses.shape[:2]
                scale_factor = (eye_dist / orig_w) * 2.6
                new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
                center_x, center_y = nx, ny - int(new_h * 0.30)

                if new_w > 0 and new_h > 0:
                    resized_filter = cv2.resize(img_glasses, (new_w, new_h))
                    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), -angle, 1.0)
                    rotated_filter = cv2.warpAffine(resized_filter, M, (new_w, new_h),
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=(0, 0, 0, 0))
                    offset_x = center_x - int(new_w / 2)
                    offset_y = center_y - int(new_h / 2)
                    img = overlay_transparent(img, rotated_filter, offset_x, offset_y)

            # ==========================================
            # 3. رسم الكاب
            # ==========================================
            if draw_hat and img_hat is not None:
                orig_h, orig_w = img_hat.shape[:2]
                face_width = math.hypot(rex - lex, rey - ley)
                scale_factor = (face_width / orig_w) * 1.5
                new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
                center_x, center_y = fx, fy - int(new_h * 0.20)

                if new_w > 0 and new_h > 0:
                    resized_filter = cv2.resize(img_hat, (new_w, new_h))
                    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), -angle, 1.0)
                    rotated_filter = cv2.warpAffine(resized_filter, M, (new_w, new_h),
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=(0, 0, 0, 0))
                    offset_x = center_x - int(new_w / 2)
                    offset_y = center_y - int(new_h / 2)
                    img = overlay_transparent(img, rotated_filter, offset_x, offset_y)

    # دمج الإطار الخارجي
    if frame_img is not None:
        frame_resized = cv2.resize(frame_img, (w_img, h_img))
        img = overlay_transparent(img, frame_resized, 0, 0)

    cv2.imshow("National Day AR Filter", img)

    # التحكم بالكيبورد كبديل للميلان
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        current_filter_key = chr(key)
        current_filter_index = filter_keys.index(current_filter_key)

cap.release()
cv2.destroyAllWindows()