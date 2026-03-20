import json
import urllib.request
import configparser
import os

import pandas as pd
import matplotlib.pyplot as plt
from urllib.error import URLError, HTTPError

CONFIG_FILE = "config.ini"
DEFAULT_API_URL = "https://www.bakubus.az/az/ajax/apiNew1"


def get_or_update_api_url() -> str:
    """
    Konfiqurasiya faylını yoxlayır, yoxdursa yaradır.
    Əgər köhnə və ya boş URL varsa, onu avtomatik yeniləyir.
    """
    config = configparser.ConfigParser()

    if not os.path.exists(CONFIG_FILE):
        config["DEFAULT"] = {"BakuBusApiUrl": DEFAULT_API_URL}
        with open(CONFIG_FILE, "w", encoding="utf-8") as configfile:
            config.write(configfile)
        return DEFAULT_API_URL

    config.read(CONFIG_FILE, encoding="utf-8")
    url = config.get("DEFAULT", "BakuBusApiUrl", fallback="").strip()

    if url == "https://www.bakubus.az/az/ajax/get-buses" or not url:
        config["DEFAULT"]["BakuBusApiUrl"] = DEFAULT_API_URL
        with open(CONFIG_FILE, "w", encoding="utf-8") as configfile:
            config.write(configfile)
        return DEFAULT_API_URL

    return url


def fetch_realtime_bus_data(url: str) -> dict:
    """
    BakuBus API-dən real vaxt telemetriya məlumatlarını əldə edir.
    """
    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Cache-Control": "no-cache",
    }

    req = urllib.request.Request(url, headers=request_headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
            print(f" Serverlə əlaqə quruldu, lakin status kodu: {response.status}")
    except HTTPError as http_err:
        print(f" HTTP Xətası: {http_err.code} - Səbəb: {http_err.reason}")
    except URLError as url_err:
        print(f" Serverə qoşulmaq mümkün olmadı: {url_err.reason}")
    except json.JSONDecodeError:
        print(" Gələn cavab etibarlı JSON formatında deyil.")
    except Exception as exc:  # noqa: BLE001
        print(f" Gözlənilməz xəta: {exc}")

    return None


def process_data_to_dataframe(payload: dict) -> pd.DataFrame:
    """
    İyerarxik JSON cavabını Pandas DataFrame-ə çevirir və tipləri tənzimləyir.
    """
    if not payload or "BUS" not in payload:
        print(" API cavabında avtobus məlumatları tapılmadı.")
        return pd.DataFrame()

    buses = payload["BUS"]

    if isinstance(buses, dict):
        buses = [buses]

    flattened_data = [bus.get("@attributes", bus) for bus in buses]
    df = pd.DataFrame(flattened_data)

    numeric_columns = ["SPEED", "LATITUDE", "LONGITUDE"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    return df


def analyze_and_visualize(df: pd.DataFrame) -> None:
    """
    Məlumatları analiz edir və Matplotlib vasitəsilə qrafik qurur.
    """
    if df.empty:
        print(" Analiz üçün məlumat yoxdur.")
        return

    print("\n--- BAKUBUS TELEMETRİYA İCMALI ---")
    print(f" Sistemdə olan aktiv avtobus sayı: {len(df)}")

    if "SPEED" in df.columns:
        active_moving = df[df["SPEED"] > 0]
        print(
            f" Hazırda hərəkətdə olan (sürəti > 0) avtobus sayı: {len(active_moving)}"
        )

        if not active_moving.empty:
            fastest_bus = active_moving.loc[active_moving["SPEED"].idxmax()]
            print("\n Ən yüksək sürətlə hərəkət edən avtobus:")
            print(
                f" ID: {fastest_bus.get('BUS_ID', 'N/A')} | "
                f" Marşrut: {fastest_bus.get('DISPLAY_ROUTE_CODE', 'N/A')} | "
                f" Sürət: {fastest_bus.get('SPEED')} km/h"
            )

    if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(
            df["LONGITUDE"],
            df["LATITUDE"],
            c=df["SPEED"] if "SPEED" in df.columns else "steelblue",
            cmap="coolwarm",
            s=40,
            alpha=0.8,
        )
        if "SPEED" in df.columns:
            plt.colorbar(sc, label="Sürət (km/h)")

        plt.title("BakuBus: Avtobusların Anlıq Mövqeyi və Sürəti")
        plt.xlabel("Uzunluq (Longitude)")
        plt.ylabel("Enlik (Latitude)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


def main() -> None:
    api_url = get_or_update_api_url()
    print(f" API-yə sorğu göndərilir: {api_url} ...")

    json_data = fetch_realtime_bus_data(api_url)

    if json_data:
        df = process_data_to_dataframe(json_data)
        analyze_and_visualize(df)
    else:
        print(" Proses dayandırıldı: Məlumat əldə edilə bilmədi.")


if __name__ == "__main__":
    main()
