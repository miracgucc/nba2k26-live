from pull_today_results import pull_today_results


def run_scraper():
    count = pull_today_results(only_tracked=False)
    print(f"Scraper finished. Added {count} completed matches.")
    return True


if __name__ == "__main__":
    run_scraper()
