import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium


# Kus cusu Mesafe Hesaplama (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


def get_predictions_and_filter(veri_yolu, model_yolu, threshold=20):
    print("Veriler ve Model yükleniyor...")
    df = pd.read_excel(veri_yolu)
    model = tf.keras.models.load_model(model_yolu)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['olcekli_doluluk'] = scaler.fit_transform(df[['doluluk_sayisal']])

    hedef_konteynerler = []

    konteyner_gruplari = df.groupby('konteyner_id')
    for konteyner_id, grup in konteyner_gruplari:
        grup = grup.sort_values(by=['tarih', 'saat'])
        son_5_veri = grup['olcekli_doluluk'].values[-5:]

        if len(son_5_veri) < 5:
            continue

        X_tahmin = np.reshape(son_5_veri, (1, 5, 1))
        tahmin_olcekli = model.predict(X_tahmin, verbose=0)

        tahmin_gercek = scaler.inverse_transform(tahmin_olcekli)[0][0]

        if tahmin_gercek >= threshold:
            enlem = grup['enlem'].iloc[-1]
            boylam = grup['boylam'].iloc[-1]
            hedef_konteynerler.append({
                'id': konteyner_id,
                'enlem': enlem,
                'boylam': boylam,
                'tahmin_doluluk': round(tahmin_gercek, 2)
            })

    print(f"\nTahmin tamamlandı! %{threshold} üzeri doluluğa ulaşacak {len(hedef_konteynerler)} konteyner bulundu.")
    return hedef_konteynerler


def create_route(konteynerler):
    if len(konteynerler) < 2:
        print("Rota oluşturmak için yeterli sayıda dolu konteyner yok.")
        return

    print("\nOR-Tools ile Rota Optimizasyonu Başlıyor...")

    num_locations = len(konteynerler)
    distance_matrix = np.zeros((num_locations, num_locations), dtype=int)

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                dist = haversine(
                    konteynerler[i]['enlem'], konteynerler[i]['boylam'],
                    konteynerler[j]['enlem'], konteynerler[j]['boylam']
                )
                distance_matrix[i][j] = int(dist)

    manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print("\n--- OPTİMİZE EDİLMİŞ ÇÖP TOPLAMA ROTASI ---")
        index = routing.Start(0)
        rota_sirasi = []
        toplam_mesafe = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            rota_sirasi.append(konteynerler[node_index])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            toplam_mesafe += routing.GetArcCostForVehicle(previous_index, index, 0)

        # Harita Gorsellestirme
        print("\nHarita oluşturuluyor...")

        baslangic_enlem = rota_sirasi[0]['enlem']
        baslangic_boylam = rota_sirasi[0]['boylam']
        harita = folium.Map(location=[baslangic_enlem, baslangic_boylam], zoom_start=15)

        koordinatlar_listesi = []

        for i, nokta in enumerate(rota_sirasi):
            print(
                f"{i + 1}. Durak: Konteyner {nokta['id']} (Tahmini Doluluk: %{nokta['tahmin_doluluk']}) -> Koordinat: {nokta['enlem']}, {nokta['boylam']}")
            koordinatlar_listesi.append([nokta['enlem'], nokta['boylam']])

            if i == 0:
                renk = 'green'  # Baslangic yeşil
                ikon = 'play'
            elif i == len(rota_sirasi) - 1:
                renk = 'red'  # Bitis kırmızı
                ikon = 'stop'
            else:
                renk = 'blue'  # Ara duraklar mavi
                ikon = 'trash'

            # Haritaya pinleri (marker) ekle
            folium.Marker(
                location=[nokta['enlem'], nokta['boylam']],
                popup=f"<b>{i + 1}. Durak</b><br>Konteyner: {nokta['id']}<br>Doluluk: %{nokta['tahmin_doluluk']}",
                icon=folium.Icon(color=renk, icon=ikon)
            ).add_to(harita)

        folium.PolyLine(
            locations=koordinatlar_listesi,
            color='red',
            weight=3,
            opacity=0.8
        ).add_to(harita)

        print(f"\nToplam Katedilecek Mesafe: {toplam_mesafe} metre")

        harita_kayit_yolu = os.path.join(os.path.dirname(__file__), '..', 'optimize_rota_haritasi.html')
        harita.save(harita_kayit_yolu)

        print(f"\n Rota başarıyla oluşturuldu!")
        print(f" Bu dosyayı tarayıcıda açarak örnek rota haritasına ulaşabilirsiniz:\n {harita_kayit_yolu}")

    else:
        print("Uygun bir rota bulunamadı!")


if __name__ == "__main__":
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'bosna_hersek_cop_verisi.xlsx')
    model_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_doluluk_modeli.keras')

    filtreli_konteynerler = get_predictions_and_filter(veri_yolu, model_yolu, threshold=20)

    create_route(filtreli_konteynerler)