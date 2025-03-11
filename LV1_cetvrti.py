
ime_datoteke = input("Unesite ime datoteke: ")

pouzdanosti = []

try:
    # Otvaranje datoteke za čitanje
    with open(ime_datoteke, 'r') as datoteka:
    # Čitanje datoteke liniju po liniju
        for linija in datoteka:
            if linija.startswith("X-DSPAM-Confidence:"):
                dioBroj = linija.split(":")[1].strip() # Dijeljenje po ":" i uklanjanje praznina
                try:
                    broj = float(dioBroj) 
                    pouzdanosti.append(broj) # Dodavanje u listu i pretvorba u float
                except ValueError:
                    # Ako nije moguće pretvoriti u broj, preskoči liniju
                    continue

    # Provjera je li lista prazna
    if len(pouzdanosti) > 0:
        # Izračun srednje vrijednosti
        srednja_pouzdanost = sum(pouzdanosti) / len(pouzdanosti)
        print(f"Srednja vrijednost pouzdanosti (X-DSPAM-Confidence): {srednja_pouzdanost}")
    else:
        print("Nema pronađenih vrijednosti pouzdanosti u datoteci.")
except FileNotFoundError:
    print(f"Greška: Datoteka '{ime_datoteke}' nije pronađena.")
