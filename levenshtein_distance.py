def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Oblicza odległość Levenshteina między dwoma napisami s1 i s2.
    """
    m, n = len(s1), len(s2)
    # Tworzymy macierz (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Inicjalizacja – koszt usunięcia/dodania znaków
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Wypełnianie macierzy
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # usunięcie
                dp[i][j - 1] + 1,      # wstawienie
                dp[i - 1][j - 1] + cost  # zamiana (jeśli różne)
            )

    return dp[m][n]
#przykład użycia
print(levenshtein_distance("energia", "energía")) 
