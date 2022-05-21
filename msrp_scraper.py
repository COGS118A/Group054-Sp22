"""Stand-alone MSRP Scraper

This program uses a multiprocess pool, which limits how functions defined here may be called.
Unexpected behavior will occur if `batch_scrape_msrp` is called in a program that has ANY code
outside of the `if __name__ == '__main__'` idiom. So far the only way I've found to work reliably
(at least on Windows) is to run this program as a CLI tool using `subprocess` in shell mode.

Beware lol. - @gbhand
"""
import argparse
from multiprocessing import Pool
import random
import re
import time

from bs4 import BeautifulSoup
import numpy as np
import requests
from requests.exceptions import HTTPError
from tqdm import tqdm

#  https://www.cars.com/research/{make}-{model}-{year}


def get_msrp_stream(car_make, car_model, car_year):
    url = f"https://www.cars.com/research/{car_make}-{car_model}-{car_year}"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if b'<div class="sds-heading--4">' in line:
                # print(line)
                break
            # print(line)
            # break

    # if r.status_code != 200:
    #     raise RuntimeError(f"Unable to locate {url}")

    # print(r.text)

    # soup = BeautifulSoup(r.text, "html.parser")

    # tag = soup.find("div", {"class": "sds-heading--4"})

    # return int(tag.text.split("$")[1].replace(",", ""))


def get_msrp_bs4(car_make, car_model, car_year):
    url = f"https://www.cars.com/research/{car_make}-{car_model}-{car_year}"
    headers = {"Range": "bytes=0-100"}  # first 100 bytes

    # with requests.get(url, stream=True) as r:
    #     r.raise_for_status()
    #     for line in r.iter_lines():
    #         if b'<div class="sds-heading--4">' in line:
    #             print(line)
    #         # print(line)
    #         # break
    r = requests.get(url)
    r.raise_for_status()
    # if r.status_code != 200:
    #     raise RuntimeError(f"Unable to locate {url}")

    # print(r.text)

    soup = BeautifulSoup(r.text, "html.parser")

    tag = soup.find("div", {"class": "sds-heading--4"})
    # print(tag)

    return int(tag.text.split("$")[1].replace(",", ""))


def get_msrp_tag(url) -> bytes:
    # raise for any response other than 200
    # not ideal, but should be faster than parsing entire incorrect page
    with requests.get(url, stream=True, allow_redirects=False) as r:
        # print(r.status_code, url)
        # r.raise_for_status()
        # print(r.history)
        if r.status_code != 200:
            raise HTTPError(f"Error: unable to locate {url}")
        for line in r.iter_lines():
            if b'<div class="sds-heading--4">' in line:
                return line


def parse_msrp_tag(tag: bytes) -> int:
    msrp = tag[34:-6]
    msrp = msrp.translate(None, b",")
    return int(msrp)


def wait_random():
    """Keep us from looking like a DDoS attack"""
    time.sleep(random.random() * 2)


def scrape_msrp(url) -> float:
    wait_random()

    try:
        tag = get_msrp_tag(url)
        # print(tag)
        msrp = parse_msrp_tag(tag)
    except HTTPError as e:
        print(e)
        return np.nan
    except TypeError:
        print(f"Error: unable to locate {url}")
        return np.nan
    print(f"got {msrp} for {url}")
    return msrp


def basic_get(car_make, car_model, car_year):
    url = f"https://www.cars.com/research/{car_make}-{car_model}-{car_year}"
    requests.get(url)


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--test", default=False, action="store_true")
    args = vars(parser.parse_args())
    return args["input"], args["output"], args["test"]


def batch_scrape_msrp(url_list: list) -> list:
    """Retrieves MSRP values from a list of urls of format https://www.cars.com/research/{car_make}-{car_model}-{car_year}

    Warning: this method uses a process pool, the calling program must be run in `if __name__ == '__main__':`

    url_list: list of urls to scrape

    Returns:
        list of MSRPs as int
    """
    MAX_THREADS = 10
    pool = Pool(processes=min(MAX_THREADS, len(url_list)))
    results = pool.map(scrape_msrp, url_list)
    return results


def generate_test_urls() -> list:
    base_url = "https://www.cars.com/research/"
    makes = {
        "honda": ["civic", "accord", "cr_v"],
        "toyota": ["corolla", "camry", "4runner"],
        "subaru": ["forester", "outback"],
        "ford": ["f_150", "f_250", "f_350"],
    }
    urls = []
    for make in makes:
        for model in makes[make]:
            urls = urls + [
                f"{base_url}{make}-{model}-{year}" for year in range(2000, 2022)
            ]

    return urls


# # st = time.time()
# # for i in tqdm(range(100)):


# print("beginning sync test")
# method = "sync"
# start_time = time.time()
# for url in all_urls:
#     scrape_msrp(url)
# avg_time = (time.time() - start_time) / n_iter

# print(f"{method} took {avg_time:.2f} seconds averaged over {n_iter} iterations")
if __name__ == "__main__":
    input_path, output_path, test_mode = handle_args()

    if test_mode:
        # testing only!
        test_urls = generate_test_urls()[:50]
        with open(input_path, mode="w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in test_urls)

    with open(input_path, mode="r", encoding="utf-8") as f:
        urls = f.read().splitlines()

    results = batch_scrape_msrp(urls)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(str(line) + "\n" for line in results)

    # print(urls)
    exit()

    # n_iter = 10
    # url = "https://www.cars.com/research/subaru-forester-200"

    # all_urls = [url + str(i) for i in range(n_iter)]
    # # for url in all_urls:
    # #     print(scrape_msrp(url))
    print("beginning async test")
    method = "async"
    urls = generate_test_urls()[:50]
    n_iter = len(urls)
    start_time = time.time()
    results = batch_scrape_msrp(urls)
    # pool = Pool(processes=n_iter)
    # results = pool.map(scrape_msrp, all_urls)
    print(results)
    # # p = Pool(n_iter)
    # # p.map(scrape_msrp, all_urls)
    # # p.terminate()
    # # p.join()
    elapsed_time = time.time() - start_time
    avg_time = (elapsed_time) / n_iter

    print(
        f"{method} took {avg_time:.2f} seconds ({elapsed_time}s total) averaged over {n_iter} iterations"
    )


# print(time.time() - st)

# while True:
# get_msrp(*input("Enter make, model, year: ").split())
# print(get_msrp(*"subaru forester 2005".split()))
# payload = "subaru forester 2005".split()
# n_iter = 100
# for i in range(3):
#     if i == 0:
#         method = "stream"
#         func = get_msrp_stream
#     elif i == 1:
#         method = "bs4"
#         func = get_msrp_bs4
#     elif i == 2:
#         method = "basic get"
#         func = basic_get

#     start_time = time.time()
#     for i in tqdm(range(n_iter)):
#         func(*payload)
#     avg_time = (time.time() - start_time) / n_iter

#     print(f"{method} took {avg_time:.2f} seconds averaged over {n_iter} iterations")
