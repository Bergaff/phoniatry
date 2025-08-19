import os
import parselmouth
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from faster_whisper import WhisperModel
from scipy.spatial import ConvexHull
import math
import re
import random
import streamlit as st

# --- Константы ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
RECTANGLE_SIZE_HZ = 1000  # Размер прямоугольника по осям F1 и F2
ENERGY_SCALE = 0.5  # Масштаб для высоты "градиента энергии" над многоугольником

os.makedirs(OUTPUT_DIR, exist_ok=True)

def transcribe_audio_with_whisper(audio_path, model_size="medium"):
    """Транскрибирует аудио с помощью Whisper, возвращая слова и их временные метки."""
    st.write(f"Загрузка модели Whisper '{model_size}'...")
    try:
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        st.write("Модель загружена. Начало транскрибации...")
        segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
        word_level_segments = [{'word': word.word.strip().lower(), 'start': word.start, 'end': word.end}
                              for segment in segments for word in segment.words if word.probability > 0.1]
        full_text = ''.join([s['word'] for s in word_level_segments])
        st.write(f"Транскрибация завершена. Распознанный текст: {full_text}")
        return word_level_segments
    except Exception as e:
        st.error(f"Ошибка при транскрибации аудио: {e}")
        return []

def extract_phonemes(text):
    """Извлекает фонемы, корректно обрабатывая йотированные гласные."""
    phonemes = []
    text_clean = re.sub(r'[^а-яё]', '', text.lower())
    for i, char in enumerate(text_clean):
        if char in 'еёюя' and (i == 0 or text_clean[i-1] not in 'аоуэыиьъ'):
            if char == 'е': phonemes.extend(['й', 'э'])
            elif char == 'ё': phonemes.extend(['й', 'о'])
            elif char == 'ю': phonemes.extend(['й', 'у'])
            elif char == 'я': phonemes.extend(['й', 'а'])
        elif char in 'еёюя':
            if char == 'е': phonemes.append('э')
            elif char == 'ё': phonemes.append('о')
            elif char == 'ю': phonemes.append('у')
            elif char == 'я': phonemes.append('а')
        elif char in 'аоуэыи': phonemes.append(char)
    return phonemes

def find_acoustic_features(formant_obj, pitch_obj, intensity_obj, segment_start, segment_end):
    """Извлекает F1, F2, F0, интенсивность и длительность для заданного временного сегмента."""
    F1_values, F2_values, pitch_values, intensity_values = [], [], [], []
    time_step = 0.005
    t = segment_start
    while t < segment_end:
        f1 = formant_obj.get_value_at_time(1, t)
        f2 = formant_obj.get_value_at_time(2, t)
        pitch = pitch_obj.get_value_at_time(t)
        intensity = intensity_obj.get_value(t)
        if not math.isnan(f1): F1_values.append(f1)
        if not math.isnan(f2): F2_values.append(f2)
        if not math.isnan(pitch): pitch_values.append(pitch)
        if not math.isnan(intensity): intensity_values.append(intensity)
        t += time_step
    median_f1 = np.nanmedian(F1_values) if F1_values else np.nan
    median_f2 = np.nanmedian(F2_values) if F2_values else np.nan
    median_pitch = np.nanmedian(pitch_values) if pitch_values else np.nan
    median_intensity = np.nanmedian(intensity_values) if intensity_values else np.nan
    duration = segment_end - segment_start
    return median_f1, median_f2, duration, median_pitch, median_intensity

def analyze_vowel_segments(audio_path, transcription_segments):
    """Анализирует гласные и возвращает акустические характеристики."""
    J_DURATION = 0.04
    vowel_data = []
    try:
        sound = parselmouth.Sound(audio_path)
        formant_obj = sound.to_formant_burg()
        pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        intensity_obj = sound.to_intensity()
    except Exception as e:
        st.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось обработать аудиофайл: {e}")
        return []
    if not transcription_segments:
        st.error("Список транскрибированных сегментов пуст.")
        return []
    for segment in transcription_segments:
        word, word_start, word_end = segment['word'], segment['start'], segment['end']
        phonemes_in_word = extract_phonemes(word)
        if not phonemes_in_word: continue
        j_count = phonemes_in_word.count('й')
        vowel_phonemes_count = len([p for p in phonemes_in_word if p != 'й'])
        effective_duration = word_end - word_start - (j_count * J_DURATION)
        if vowel_phonemes_count == 0 or effective_duration <= 0: continue
        vowel_duration_part = effective_duration / vowel_phonemes_count
        current_time = word_start
        for phoneme in phonemes_in_word:
            if phoneme == 'й':
                current_time += J_DURATION
                continue
            vowel_segment_start = current_time
            vowel_segment_end = current_time + vowel_duration_part
            median_f1, median_f2, duration, median_pitch, median_intensity = find_acoustic_features(
                formant_obj, pitch_obj, intensity_obj, vowel_segment_start, vowel_segment_end
            )
            if not (math.isnan(median_f1) or math.isnan(median_f2) or math.isnan(duration) or math.isnan(median_pitch)):
                impulses = median_pitch * duration
                total_energy = 0.00012 * impulses - 0.00015
                vowel_data.append({
                    'word': word, 'vowel': phoneme, 'F1': median_f1, 'F2': median_f2,
                    'duration': duration, 'mean_pitch': median_pitch, 'mean_intensity': median_intensity,
                    'start_time': vowel_segment_start, 'end_time': vowel_segment_end, 'total_energy': total_energy
                })
                st.write(f"Фонема '{phoneme}' в слове '{word}': F1={median_f1:.2f}, F2={median_f2:.2f}, Длит-сть={duration:.2f}, Тон={median_pitch:.2f} Гц, Интенсивность={median_intensity:.2f} дБ, Энергия={total_energy:.6f} Pa^2·sec")
            current_time = vowel_segment_end
    st.write(f"Всего данных о гласных собрано: {len(vowel_data)}")
    return vowel_data

def plot_vowel_histogram(vowel_data):
    """Строит гистограмму количества гласных."""
    if not vowel_data:
        st.error("Нет данных для построения гистограммы.")
        return None
    df = pd.DataFrame(vowel_data)
    vowel_counts = df['vowel'].value_counts().reset_index()
    vowel_counts.columns = ['vowel', 'count']
    
    fig = px.histogram(vowel_counts, x='vowel', y='count', title='Распределение гласных',
                      labels={'vowel': 'Гласная', 'count': 'Количество'},
                      color='vowel', color_discrete_map={'и': 'blue', 'э': 'green', 'а': 'yellow', 'о': 'orange', 'у': 'purple', 'ы': 'pink'})
    fig.update_layout(width=800, height=600, showlegend=True)
    return fig

def are_points_collinear(points):
    """Проверяет, являются ли точки коллинеарными."""
    if len(points) < 3:
        return True
    points = np.array(points)
    # Проверяем, лежат ли все точки на одной прямой, вычисляя ранг матрицы
    matrix = points - points[0]
    rank = np.linalg.matrix_rank(matrix)
    return rank < 2

def plot_3d_with_polygons(vowel_data, audio_filename):
    """Строит 3D-график с нормализованной длительностью по оси Z, многоугольниками на основе F1 и F2."""
    base_name = os.path.splitext(os.path.basename(audio_filename))[0]
    if not vowel_data:
        st.error("Нет данных для построения графика.")
        return None
    df = pd.DataFrame(vowel_data)
    fig = go.Figure()
    vowel_colors = {'и': 'blue', 'э': 'green', 'а': 'yellow', 'о': 'orange', 'у': 'purple', 'ы': 'pink'}

    # Нормализация данных
    max_duration = df['duration'].max()
    min_duration = df['duration'].min()
    max_pitch = df['mean_pitch'].max()
    min_pitch = df['mean_pitch'].min()
    max_energy = df['total_energy'].max()
    min_energy = df['total_energy'].min()
    duration_range = max_duration - min_duration if max_duration != min_duration else 1
    pitch_range = max_pitch - min_pitch if max_pitch != min_pitch else 1
    energy_range = max_energy - min_energy if max_energy != min_energy else 1

    df['norm_duration'] = (df['duration'] - min_duration) / duration_range
    df['log_pitch'] = df['mean_pitch'].apply(lambda x: np.log(max(x, 1)))
    max_log_pitch = df['log_pitch'].max()
    min_log_pitch = df['log_pitch'].min()
    log_pitch_range = max_log_pitch - min_log_pitch if max_log_pitch != min_log_pitch else 1
    df['norm_log_pitch'] = (df['log_pitch'] - min_log_pitch) / log_pitch_range
    df['norm_energy'] = (df['total_energy'] - min_energy) / energy_range

    duration_scale = 10.0
    max_scaled_duration = 1.0 * duration_scale

    for vowel, group in df.groupby('vowel'):
        color = vowel_colors.get(vowel, 'gray')
        if len(group) < 1:
            continue

        max_duration_value = group['duration'].max()
        max_duration_rows = group[group['duration'] == max_duration_value]
        highest_point = max_duration_rows.sample(n=1, random_state=random.randint(0, 1000)).iloc[0] if len(max_duration_rows) > 1 else max_duration_rows.iloc[0]

        st.write(f"Фонема '{vowel}' (высшая точка по длительности):")
        st.write(f"  Тон (mean_pitch): {highest_point['mean_pitch']:.2f} Гц")
        st.write(f"  Логарифм тона (log_pitch): {highest_point['log_pitch']:.4f}")
        st.write(f"  Нормализованный логарифм тона (norm_log_pitch): {highest_point['norm_log_pitch']:.4f}")
        st.write(f"  Энергия (total_energy): {highest_point['total_energy']:.6f} Pa^2·sec")
        st.write(f"  Нормализованная энергия (norm_energy): {highest_point['norm_energy']:.4f}")

        center_x = highest_point['F1']
        center_y = highest_point['F2']
        norm_log_pitch = highest_point['norm_log_pitch']
        norm_duration = highest_point['norm_duration'] * duration_scale
        plane_z = norm_duration - (norm_log_pitch * max_scaled_duration)
        points_2d = group[['F1', 'F2']].values

        if len(group) >= 3:
            try:
                hull_2d = ConvexHull(points_2d)
                area = hull_2d.volume
                hull_vertices = points_2d[hull_2d.vertices]
                hull_x = hull_vertices[:, 0].tolist()
                hull_y = hull_vertices[:, 1].tolist()
                hull_x.append(hull_x[0])
                hull_y.append(hull_y[0])
                hull_z = [plane_z] * len(hull_x)

                fig.add_trace(go.Scatter3d(
                    x=hull_x, y=hull_y, z=hull_z,
                    mode='lines', line=dict(color=color, width=5),
                    name=f'Плоскость "{vowel}" (Площадь: {area:.2f})',
                    hoverinfo='name', showlegend=True
                ))

                section_x = hull_x[:-1] + [center_x, hull_x[0]]
                section_y = hull_y[:-1] + [center_y, hull_y[0]]
                section_z = [norm_duration] * (len(hull_x) - 1) + [norm_duration, plane_z]
                section_i = list(range(len(hull_x) - 1)) + [0]
                section_j = list(range(1, len(hull_x))) + [len(hull_x) - 1]
                section_k = [len(hull_x) - 1] * (len(hull_x) - 1) + [0]

                fig.add_trace(go.Mesh3d(
                    x=section_x, y=section_y, z=section_z,
                    i=section_i, j=section_j, k=section_k,
                    color=color, opacity=0.3,
                    name=f'Секция "{vowel}"',
                    hoverinfo='name', showlegend=True, visible='legendonly'
                ))

                centroid = group[['F1', 'F2']].mean()
                group['angle'] = group.apply(lambda row: np.arctan2(row['F2'] - centroid['F2'], row['F1'] - centroid['F1']), axis=1)
                group_sorted = group.sort_values(by='angle', ascending=False)

                x_coords = group_sorted['F1'].tolist()
                y_coords = group_sorted['F2'].tolist()
                z_coords = (group_sorted['norm_duration'] * duration_scale).tolist()
                n_points = len(x_coords)

                energy_values = [0.00012 * (row['mean_pitch'] * row['duration']) - 0.00015 for _, row in group_sorted.iterrows()]
                max_energy_vowel = max(energy_values) if energy_values else 1.0
                min_energy_vowel = min(energy_values) if energy_values else 0.0
                energy_range_vowel = max_energy_vowel - min_energy_vowel if max_energy_vowel != min_energy_vowel else 1.0
                norm_energy_values = [(e - min_energy_vowel) / energy_range_vowel for e in energy_values]

                i_list, j_list, k_list = [], [], []
                for i in range(n_points - 2):
                    i_list.extend([0, i + 1])
                    j_list.extend([i + 1, i + 2])
                    k_list.extend([i + 2, 0])

                fig.add_trace(go.Mesh3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    i=i_list, j=j_list, k=k_list,
                    color=color, opacity=0.3,
                    name=f'Область "{vowel}"',
                    hoverinfo='name', showlegend=True, visible='legendonly'
                ))

                below_x, below_y, below_z = [], [], []
                for x, y, z in zip(x_coords, y_coords, z_coords):
                    if z <= plane_z:
                        below_x.append(x)
                        below_y.append(y)
                        below_z.append(z)
                    else:
                        below_x.append(x)
                        below_y.append(y)
                        below_z.append(plane_z)

                if len(below_x) >= 3:
                    below_points_2d = np.array(list(zip(below_x, below_y)))
                    unique_points = np.unique(below_points_2d, axis=0)
                    if len(unique_points) >= 3 and not are_points_collinear(unique_points):
                        try:
                            below_hull = ConvexHull(below_points_2d)
                            below_i = below_hull.simplices[:, 0]
                            below_j = below_hull.simplices[:, 1]
                            below_k = below_hull.simplices[:, 2]
                            fig.add_trace(go.Mesh3d(
                                x=below_x, y=below_y, z=below_z,
                                i=below_i, j=below_j, k=below_k,
                                color=color, opacity=0.3,
                                name=f'Область ниже плоскости "{vowel}"',
                                hoverinfo='name', showlegend=True, visible='legendonly'
                            ))
                        except Exception as e:
                            st.warning(f"Не удалось построить область ниже плоскости для фонемы '{vowel}': {e}")
                    else:
                        st.warning(f"Пропуск области ниже плоскости для фонемы '{vowel}': недостаточно уникальных или неколлинеарных точек ({len(unique_points)} точек).")
                else:
                    st.warning(f"Пропуск области ниже плоскости для фонемы '{vowel}': недостаточно точек ({len(below_x)}).")

                proj_x = x_coords[:]
                proj_y = y_coords[:]
                proj_z = [plane_z] * len(x_coords)
                fig.add_trace(go.Mesh3d(
                    x=proj_x, y=proj_y, z=proj_z,
                    i=i_list, j=j_list, k=k_list,
                    color=color, opacity=0.5,
                    name=f'Проекция области на плоскости "{vowel}"',
                    hoverinfo='name', showlegend=True, visible='legendonly'
                ))

                base_color = vowel_colors.get(vowel, 'gray')
                color_map = {
                    'blue': (0, 0, 255), 'green': (0, 128, 0), 'yellow': (255, 255, 0),
                    'orange': (255, 165, 0), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'gray': (128, 128, 128)
                }
                base_rgb = color_map.get(base_color, (128, 128, 128))
                custom_colorscale = [
                    [0, f'rgb({int(0 + 0.3 * base_rgb[0])},{int(0 + 0.3 * base_rgb[1])},{int(255 * 0.7 + 0.3 * base_rgb[2])})'],
                    [1, f'rgb({int(255 * 0.7 + 0.3 * base_rgb[0])},{int(0 + 0.3 * base_rgb[1])},{int(0 + 0.3 * base_rgb[2])})']
                ]

                energy_z = [z + norm_energy * ENERGY_SCALE for z, norm_energy in zip(z_coords, norm_energy_values)]
                energy_intensity = norm_energy_values

                fig.add_trace(go.Mesh3d(
                    x=x_coords, y=y_coords, z=energy_z,
                    i=i_list, j=j_list, k=k_list,
                    intensity=energy_intensity, colorscale=custom_colorscale, showscale=False,
                    opacity=0.5, name=f'Градиент энергии "{vowel}"',
                    hoverinfo='text',
                    hovertext=[f'Фонема: {vowel}<br>F1: {x:.2f}<br>F2: {y:.2f}<br>Энергия: {e:.6f}<br>Норм. энергия: {ne:.2f}'
                               for x, y, e, ne in zip(x_coords, y_coords, energy_values, norm_energy_values)],
                    showlegend=True, visible='legendonly'
                ))

                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=energy_z,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=energy_intensity,
                        colorscale=custom_colorscale,
                        showscale=False,
                        opacity=0.8
                    ),
                    name=f'Пики градиента энергии "{vowel}"',
                    hoverinfo='text',
                    hovertext=[f'Фонема: {vowel}<br>F1: {x:.2f}<br>F2: {y:.2f}<br>Энергия: {e:.6f}<br>Норм. энергия: {ne:.2f}'
                               for x, y, e, ne in zip(x_coords, y_coords, energy_values, norm_energy_values)],
                    showlegend=True, visible='legendonly'
                ))

            except Exception as e:
                st.warning(f"Не удалось построить многоугольник для фонемы '{vowel}': {e}")

        x_coords_line = [center_x, center_x]
        y_coords_line = [center_y, center_y]
        z_coords_line = [norm_duration, plane_z]
        fig.add_trace(go.Scatter3d(
            x=x_coords_line, y=y_coords_line, z=z_coords_line,
            mode='lines', line=dict(color=color, width=5, dash='dash'),
            name=f'Спуск "{vowel}"',
            hoverinfo='text',
            hovertext=f'Фонема: {vowel}<br>F1: {center_x:.2f}<br>F2: {center_y:.2f}<br>Норм. лог. тон: {norm_log_pitch:.2f}',
            showlegend=True
        ))

        if len(group) > 1:
            x_coords = group_sorted['F1'].tolist()
            y_coords = group_sorted['F2'].tolist()
            z_coords = (group_sorted['norm_duration'] * duration_scale).tolist()
            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])
            z_coords.append(z_coords[0])
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines', line=dict(color=color, width=5),
                name=f'Линии "{vowel}"',
                showlegend=True, hoverinfo='text',
                hovertext=group_sorted.apply(lambda row: f'Фонема: {row["vowel"]}<br>F1: {row["F1"]:.2f}<br>F2: {row["F2"]:.2f}<br>Норм. длительность: {row["norm_duration"]:.2f}', axis=1)
            ))

    visibility_states_polygons = []
    visibility_states_sections = []
    visibility_states_below = []
    visibility_states_projection = []
    visibility_states_energy = []
    for i, trace in enumerate(fig.data):
        if 'Область "' in trace.name and 'ниже' not in trace.name and 'проекция' not in trace.name.lower() and 'энергия' not in trace.name.lower():
            visibility_states_polygons.append(i)
        if 'Секция' in trace.name:
            visibility_states_sections.append(i)
        if 'Область ниже плоскости' in trace.name:
            visibility_states_below.append(i)
        if 'Проекция области' in trace.name:
            visibility_states_projection.append(i)
        if 'Градиент энергии' in trace.name or 'Пики градиента энергии' in trace.name:
            visibility_states_energy.append(i)

    buttons = [
        dict(label="Показать всё",
             method="update",
             args=[{"visible": [True] * len(fig.data)},
                   {"title": f'3D-карта гласных - {base_name} (Все показаны)'}]),
        dict(label="Скрыть всё",
             method="update",
             args=[{"visible": ['legendonly'] * len(fig.data)},
                   {"title": f'3D-карта гласных - {base_name} (Все скрыто, легенда активна)'}]),
        dict(label="Скрыть многоугольники и секции",
             method="update",
             args=[{"visible": ['legendonly' if i in visibility_states_polygons or i in visibility_states_sections else True for i in range(len(fig.data))]},
                   {"title": f'3D-карта гласных - {base_name} (Многоугольники и секции скрыты)'}]),
        dict(label="Показать область ниже плоскости",
             method="update",
             args=[{"visible": [True if i in visibility_states_below else 'legendonly' for i in range(len(fig.data))]},
                   {"title": f'3D-карта гласных - {base_name} (Область ниже плоскости показана)'}]),
        dict(label="Показать проекцию на плоскости",
             method="update",
             args=[{"visible": [True if i in visibility_states_projection else 'legendonly' for i in range(len(fig.data))]},
                   {"title": f'3D-карта гласных - {base_name} (Проекция на плоскости показана)'}]),
        dict(label="Показать градиенты энергии",
             method="update",
             args=[{"visible": [True if i in visibility_states_energy else 'legendonly' for i in range(len(fig.data))]},
                   {"title": f'3D-карта гласных - {base_name} (Градиенты энергии показаны)'}])
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=-0.1,
                xanchor="center",
                yanchor="top",
                buttons=buttons
            )
        ],
        title=f'3D-карта гласных (норм. длительность по Z) - {base_name}',
        scene=dict(
            xaxis_title='F1 (Гц)', yaxis_title='F2 (Гц)', zaxis_title='Норм. длительность (с)',
            xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"), zaxis=dict(range=[0, max_scaled_duration * 1.2]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
        width=1000, height=800, showlegend=True
    )
    return fig

def main():
    st.title("Анализ и визуализация гласных в аудио")
    uploaded_file = st.file_uploader("Выберите WAV-аудиофайл", type=["wav"])

    if uploaded_file is not None:
        audio_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("\nНачинаем транскрибацию аудио...")
        transcription_segments = transcribe_audio_with_whisper(audio_path, model_size=WHISPER_MODEL)
        
        if transcription_segments:
            st.write("\nНачинаем акустический анализ для сбора данных о формантах, длительности и тоне...")
            vowel_data = analyze_vowel_segments(audio_path, transcription_segments)
            
            if vowel_data:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                csv_path = os.path.join(OUTPUT_DIR, f'{base_name}_vowel_formants_params_raw.csv')
                pd.DataFrame(vowel_data).to_csv(csv_path, index=False, float_format='%.4f')
                st.write(f"\nВсе измерения сохранены в: {csv_path}")
                
                # Построение гистограммы
                st.subheader("Гистограмма количества гласных")
                hist_fig = plot_vowel_histogram(vowel_data)
                if hist_fig:
                    st.plotly_chart(hist_fig)
                
                # Построение 3D-графика
                st.subheader("3D-карта гласных")
                fig_3d = plot_3d_with_polygons(vowel_data, audio_path)
                if fig_3d:
                    st.plotly_chart(fig_3d)
                
                html_path = os.path.join(OUTPUT_DIR, f"{base_name}_vowel_discrete.html")
                fig_3d.write_html(html_path)
                st.write(f"\nИнтерактивный график с многоугольниками и срезами сохранен в: {html_path}")
            else:
                st.error("\nНе удалось извлечь данные о гласных для анализа.")
        else:
            st.error("\nНе удалось транскрибировать аудио.")
    else:
        st.info("Пожалуйста, загрузите WAV-аудиофайл для анализа.")

if __name__ == "__main__":
    main()
