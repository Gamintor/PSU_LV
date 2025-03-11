try: 
    unos = float(input("Unesite broj između 0.0 i 1.0"))
    if 0.0 <= unos <= 1.0:
        if unos >= 0.9:
            kategorija = 'A'
        elif unos >= 0.8:
            kategorija = 'B'
        elif unos >= 0.7:
            kategorija = 'C'
        elif unos >= 0.6:
            kategorija = 'D'
        else: 
            kategorija = 'F'
        print(f"Ocjena {unos} pripada kategoriji {kategorija}")
    else:
        print("Uneseni broj nije unutar intervala!")    
except ValueError:
    print("Nije unesen broj")