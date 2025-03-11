ime_datoteke = "song.txt"

brojac_rijeci = {}

try:
    with open(ime_datoteke, 'r') as datoteka:
        # Čitanje datoteke liniju po liniju
        for linija in datoteka:
            # Uklanjanje znakova za novi red i pretvaranje u mala slova
            linija = linija.strip().lower()
            rijeci = linija.split()     # Razdvajanje linije na riječi
            # Brojanje riječi
            for rijec in rijeci:
                # Ako riječ već postoji, povećaj brojač
                if rijec in brojac_rijeci:
                    brojac_rijeci[rijec] += 1
                # Inače, dodaj riječ u rječnik s brojačem 1
                else:
                    brojac_rijeci[rijec] = 1

    # Pronalaženje riječi koje se pojavljuju samo jednom
    rijeci_jednom = [rijec for rijec, broj in brojac_rijeci.items() if broj == 1]

    # Ispis rezultata
    print(f"Ukupan broj različitih riječi: {len(brojac_rijeci)}")
    print(f"Broj riječi koje se pojavljuju samo jednom: {len(rijeci_jednom)}")
    print("Riječi koje se pojavljuju samo jednom:")
    for rijec in rijeci_jednom:
        print(rijec)

except FileNotFoundError:
    print(f"Greška: Datoteka '{ime_datoteke}' nije pronađena.")
except Exception as e:
    print(f"Greška: Došlo je do problema pri čitanju datoteke: {e}")