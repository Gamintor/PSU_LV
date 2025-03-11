brojevi = []

while True:
    unos = input("Unesite broj ili \"done\"")

    if unos.lower() == "done":
       break                    #prekid petlje

    try:                        #Castanje unosa u float te dodavanje u listu
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("Nije unesen broj!")
        
if len(brojevi) > 0:            #Provjera jesmo li što unijeli u listu prije računanja
    brojUnesenih = len(brojevi)
    srednjaVrijednost = sum(brojevi) / brojUnesenih
    minVrijednost = min(brojevi)
    maxVrijednost = max(brojevi)
    brojevi.sort()

    print(f"Broj unesenih brojeva: {brojUnesenih}")
    print(f"Srednja Vrijednost: {srednjaVrijednost}")
    print(f"Minimalna Vrijednost: {minVrijednost}")
    print(f"Srednja Vrijednost: {srednjaVrijednost}")
    print(f"Sortirana lista: {brojevi}")
else:
    print("Niste unijeli nijedan broj!")