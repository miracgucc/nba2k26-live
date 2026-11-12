import csv
import random
from datetime import datetime, timedelta

# CyberLeague Takımları ve Güç Seviyeleri (70-95 arası)
TEAMS = {
    "Boston Celtics (cyber)": 94,
    "Golden State Warriors (cyber)": 92,
    "Los Angeles Lakers (cyber)": 89,
    "Milwaukee Bucks (cyber)": 91,
    "Phoenix Suns (cyber)": 88,
    "Denver Nuggets (cyber)": 93,
    "Philadelphia 76ers (cyber)": 87,
    "Dallas Mavericks (cyber)": 90,
    "Miami Heat (cyber)": 85,
    "Los Angeles Clippers (cyber)": 86,
    "Chicago Bulls (cyber)": 82,
    "Houston Rockets (cyber)": 80,
    "New York Knicks (cyber)": 86,
    "Brooklyn Nets (cyber)": 78,
    "Memphis Grizzlies (cyber)": 84,
    "Sacramento Kings (cyber)": 85,
    "Indiana Pacers (cyber)": 83,
    "Cleveland Cavaliers (cyber)": 88,
    "Minnesota Timberwolves (cyber)": 89,
    "Oklahoma City Thunder (cyber)": 90,
    "New Orleans Pelicans (cyber)": 83,
    "Atlanta Hawks (cyber)": 81,
    "Orlando Magic (cyber)": 84,
    "Toronto Raptors (cyber)": 79,
    "Portland Trail Blazers (cyber)": 77,
    "Utah Jazz (cyber)": 76,
    "San Antonio Spurs (cyber)": 80,
    "Detroit Pistons (cyber)": 74,
    "Charlotte Hornets (cyber)": 75,
    "Washington Wizards (cyber)": 73
}

# Takımların Cyber Yıldız Oyuncuları (İstatistiksel derinlik için)
TEAM_STARS = {
    "Boston Celtics (cyber)": ["Tatum Cyber", "Brown Cyber", "White Cyber"],
    "Golden State Warriors (cyber)": ["Steph Cyber", "Green Cyber", "Kuminga Cyber"],
    "Los Angeles Lakers (cyber)": ["LeBron Cyber", "AD Cyber", "Reaves Cyber"],
    "Milwaukee Bucks (cyber)": ["Giannis Cyber", "Lillard Cyber", "Middleton Cyber"],
    "Phoenix Suns (cyber)": ["KD Cyber", "Booker Cyber", "Beal Cyber"],
    "Denver Nuggets (cyber)": ["Jokic Cyber", "Murray Cyber", "Porter Cyber"],
    "Philadelphia 76ers (cyber)": ["Embiid Cyber", "George Cyber", "Maxey Cyber"],
    "Dallas Mavericks (cyber)": ["Luka Cyber", "Kyrie Cyber", "Thompson Cyber"],
    "Miami Heat (cyber)": ["Butler Cyber", "Adebayo Cyber", "Herro Cyber"],
    "Los Angeles Clippers (cyber)": ["Kawhi Cyber", "Harden Cyber", "Zubac Cyber"],
    "Chicago Bulls (cyber)": ["LaVine Cyber", "Vucevic Cyber", "White Cyber"],
    "Houston Rockets (cyber)": ["Sengun Cyber", "Green Cyber", "VanVleet Cyber"],
    "New York Knicks (cyber)": ["Brunson Cyber", "Towns Cyber", "Anunoby Cyber"],
    "Brooklyn Nets (cyber)": ["Thomas Cyber", "Schroder Cyber", "Claxton Cyber"],
    "Memphis Grizzlies (cyber)": ["Ja Cyber", "Bane Cyber", "Jackson Cyber"],
    "Sacramento Kings (cyber)": ["Fox Cyber", "Sabonis Cyber", "DeRozan Cyber"],
    "Indiana Pacers (cyber)": ["Haliburton Cyber", "Siakam Cyber", "Turner Cyber"],
    "Cleveland Cavaliers (cyber)": ["Mitchell Cyber", "Garland Cyber", "Allen Cyber"],
    "Minnesota Timberwolves (cyber)": ["Edwards Cyber", "Gobert Cyber", "Conley Cyber"],
    "Oklahoma City Thunder (cyber)": ["SGA Cyber", "Chet Cyber", "Jalen Cyber"],
    "New Orleans Pelicans (cyber)": ["Zion Cyber", "Ingram Cyber", "CJ Cyber"],
    "Atlanta Hawks (cyber)": ["Young Cyber", "Johnson Cyber", "Capela Cyber"],
    "Orlando Magic (cyber)": ["Banchero Cyber", "Wagner Cyber", "Suggs Cyber"],
    "Toronto Raptors (cyber)": ["Barnes Cyber", "Barrett Cyber", "Quickley Cyber"],
    "Portland Trail Blazers (cyber)": ["Grant Cyber", "Ayton Cyber", "Simons Cyber"],
    "Utah Jazz (cyber)": ["Markkanen Cyber", "Sexton Cyber", "Collins Cyber"],
    "San Antonio Spurs (cyber)": ["Wemby Cyber", "Paul Cyber", "Vassell Cyber"],
    "Detroit Pistons (cyber)": ["Cade Cyber", "Harris Cyber", "Duren Cyber"],
    "Charlotte Hornets (cyber)": ["LaMelo Cyber", "Miller Cyber", "Miles Cyber"],
    "Washington Wizards (cyber)": ["Poole Cyber", "Kuzma Cyber", "Valanciunas Cyber"]
}

def generate_quarter_score(rating, opponent_rating, is_home):
    # Temel periyot skoru normal dağılımla üretilir
    # Ev sahibi avantajı +1.5 sayı ortalamaya eklenir
    base_mean = 26.5
    rating_diff = rating - opponent_rating
    
    mean = base_mean + (rating_diff * 0.12)
    if is_home:
        mean += 1.2
        
    # Standart sapma ile gerçekçi dalgalanma sağlanır
    score = int(random.normalvariate(mean, 4.2))
    return max(15, min(score, 45)) # Skor sınırları (15-45 arası)

def generate_match_data(num_matches=2000):
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    end_date = datetime(2026, 7, 18, 15, 0, 0)
    time_delta_seconds = int((end_date - start_date).total_seconds())
    
    matches = []
    team_list = list(TEAMS.keys())
    
    for i in range(num_matches):
        # Rastgele tarih oluştur (Ocak 2026 - Temmuz 2026 arası)
        random_seconds = random.randint(0, time_delta_seconds)
        match_time = start_date + timedelta(seconds=random_seconds)
        
        # Ev sahibi ve deplasman takımlarını seç
        home_team = random.choice(team_list)
        away_team = random.choice(team_list)
        while away_team == home_team:
            away_team = random.choice(team_list)
            
        home_rating = TEAMS[home_team]
        away_rating = TEAMS[away_team]
        
        # Periyot skorlarını üret
        hq1 = generate_quarter_score(home_rating, away_rating, True)
        aq1 = generate_quarter_score(away_rating, home_rating, False)
        
        hq2 = generate_quarter_score(home_rating, away_rating, True)
        aq2 = generate_quarter_score(away_rating, home_rating, False)
        
        hq3 = generate_quarter_score(home_rating, away_rating, True)
        aq3 = generate_quarter_score(away_rating, home_rating, False)
        
        hq4 = generate_quarter_score(home_rating, away_rating, True)
        aq4 = generate_quarter_score(away_rating, home_rating, False)
        
        home_total = hq1 + hq2 + hq3 + hq4
        away_total = aq1 + aq2 + aq3 + aq4
        
        ot_home = 0
        ot_away = 0
        is_ot = False
        
        # Uzatma periyodu (Beraberlik durumunda)
        if home_total == away_total:
            is_ot = True
            ot_home = random.randint(5, 18)
            ot_away = random.randint(5, 18)
            while ot_home == ot_away:
                ot_home = random.randint(5, 18)
                ot_away = random.randint(5, 18)
            home_total += ot_home
            away_total += ot_away
            
        winner = home_team if home_total > away_total else away_team
        total_score = home_total + away_total
        margin = home_total - away_total # Pozitif ise Ev sahibi önde, negatif ise Deplasman önde
        
        # Yıldız oyuncu performansı (En skorer oyuncu)
        winning_team = home_team if winner == home_team else away_team
        star_player = random.choice(TEAM_STARS[winning_team])
        star_points = int(random.normalvariate(28, 5))
        star_points = max(18, min(star_points, 52))
        
        matches.append({
            "game_id": f"CB-{100000 + i}",
            "date": match_time.strftime("%Y-%m-%d %H:%M"),
            "home_team": home_team,
            "away_team": away_team,
            "home_q1": hq1,
            "away_q1": aq1,
            "home_q2": hq2,
            "away_q2": aq2,
            "home_q3": hq3,
            "away_q3": aq3,
            "home_q4": hq4,
            "away_q4": aq4,
            "ot_home": ot_home,
            "ot_away": ot_away,
            "home_total": home_total,
            "away_total": away_total,
            "winner": winner,
            "total_score": total_score,
            "margin": margin,
            "is_ot": 1 if is_ot else 0,
            "top_scorer": star_player,
            "top_scorer_points": star_points
        })
        
    # Maçları tarihe göre sırala
    matches.sort(key=lambda x: x["date"])
    
    # CSV dosyasına yaz
    fields = [
        "game_id", "date", "home_team", "away_team",
        "home_q1", "away_q1", "home_q2", "away_q2",
        "home_q3", "away_q3", "home_q4", "away_q4",
        "ot_home", "ot_away", "home_total", "away_total",
        "winner", "total_score", "margin", "is_ot",
        "top_scorer", "top_scorer_points"
    ]
    
    csv_file = "matches_data.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matches)
        
    print(f"Başarıyla {num_matches} adet maç verisi üretildi ve '{csv_file}' dosyasına kaydedildi.")

if __name__ == "__main__":
    generate_match_data()
