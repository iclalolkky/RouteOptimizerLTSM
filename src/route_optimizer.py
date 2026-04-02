import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import requests
import math


def get_osrm_distance_matrix(konteynerler):
    coords = ";".join([f"{k['boylam']},{k['enlem']}" for k in konteynerler])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            matrix = np.array(data['distances'], dtype=int)
            return matrix
        else:
            print(f"OSRM API Hatası (Kod: {response.status_code}).")
            return None
    except Exception as e:
        print(f"Bağlantı hatası: {e}")
        return None


def get_predictions_and_filter(veri_yolu, model_yolu, threshold=45):
    print("Sistem Başlatılıyor: Veriler ve Model yükleniyor...")
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

    print(
        f"Tahmin tamamlandı! %{threshold} üzeri doluluğa ulaşacak toplam {len(hedef_konteynerler)} konteyner bulundu.")
    return hedef_konteynerler


def vardiyalara_bol(konteynerler):
    depo = konteynerler[0]
    kalanlar = konteynerler[1:]

    if len(kalanlar) < 2:
        return [depo] + kalanlar, [depo]

    # K-Means ile cografik kume: ayni bolgedeki konteynerleri ayni vardiyaya at
    koordinatlar = np.array([[k['enlem'], k['boylam']] for k in kalanlar])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    etiketler = kmeans.fit_predict(koordinatlar)

    sabah_vardiyasi = [depo] + [k for k, e in zip(kalanlar, etiketler) if e == 0]
    aksam_vardiyasi = [depo] + [k for k, e in zip(kalanlar, etiketler) if e == 1]

    return sabah_vardiyasi, aksam_vardiyasi


def create_route(konteynerler, vardiya_adi):
    if len(konteynerler) < 5:
        print(f"\n[!] {vardiya_adi} için yeterli sayıda dolu konteyner yok.")
        return

    print(f"\n{'=' * 60}")
    print(f" {vardiya_adi.upper()} ROTA RAPORU (5 KAMYON)")
    print(f"{'=' * 60}")
    print("OSRM üzerinden karayolu mesafeleri hesaplanıyor...")

    num_locations = len(konteynerler)
    num_vehicles = 5
    depot = 0

    distance_matrix = get_osrm_distance_matrix(konteynerler)

    if distance_matrix is None:
        print("HATA: Karayolu verisi alınamadı. Rota iptal edildi.")
        return

    print("Yol verisi başarıyla çekildi. Rota Optimizasyonu başlatılıyor (10 sn sürebilir)...")

    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Mesafe dengesi: her kamyonun katedecegi km'yi esitlemeye calis
    # SetGlobalSpanCostCoefficient yuksek tutulursa km farki minimize edilir
    max_arac_mesafesi = 150_000  # kamyon basi max 150 km
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,
        max_arac_mesafesi,
        True,
        dimension_name
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(10_000)  # yuksek katsayi = daha katı km dengesi

    # Kapasite: doluluk yuzdesi taban alinir (hacim bazli esit yuk)
    # Her konteynerin "agirlik" talebi tahmin edilen doluluk orani
    demands = [0] + [max(1, int(k['tahmin_doluluk'])) for k in konteynerler[1:]]
    toplam_talep = sum(demands)
    max_kapasite = math.ceil(toplam_talep / num_vehicles) + 15  # hafif tampon

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimension(
        demand_callback_index,
        0,
        max_kapasite,
        True,
        'Capacity'
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        toplam_filo_mesafesi = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            rota_guzergahi = []
            arac_mesafesi = 0
            arac_yuzdoluluk = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index == 0:
                    rota_guzergahi.append("DEPO")
                else:
                    rota_guzergahi.append(konteynerler[node_index]['id'])
                    arac_yuzdoluluk += demands[node_index]

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                arac_mesafesi += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            rota_guzergahi.append("DEPO")
            toplam_filo_mesafesi += arac_mesafesi

            if len(rota_guzergahi) > 2:
                arac_km = round(arac_mesafesi / 1000, 1)
                print(f"\n  Kamyon {vehicle_id + 1}:")
                print(f"    Konteyner Sayisi : {len(rota_guzergahi) - 2} adet")
                print(f"    Toplam Doluluk   : {arac_yuzdoluluk} birim  (esit yuk dengesi)")
                print(f"    Katedilen Mesafe : {arac_km} km")
                print(f"    Guzergah         : {' -> '.join(rota_guzergahi)}")

        toplam_km = round(toplam_filo_mesafesi / 1000, 1)
        print(f"\n  Toplam Filo Mesafesi: {toplam_km} km")

    else:
        print("Uygun bir rota bulunamadı! (Matematiksel olarak çözülemedi)")


if __name__ == "__main__":
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'bosna_hersek_cop_verisi_gercekci.xlsx')
    model_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_doluluk_modeli.keras')

    filtreli_konteynerler = get_predictions_and_filter(veri_yolu, model_yolu, threshold=45)
    sabah_listesi, aksam_listesi = vardiyalara_bol(filtreli_konteynerler)

    create_route(sabah_listesi, "Sabah Vardiyası")
    create_route(aksam_listesi, "Akşam Vardiyası")