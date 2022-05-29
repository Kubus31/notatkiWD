import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##############NUMPY

# print(b.shape)  # Wyswietlanie ksztaltu macierzy
# print(b.dtype)  #'Pokazuje ilu bitowa jest macierz'
#
# #otrzymywanie wymiarow macierzy
# print(a.ndim)
# print(b.ndim)
#
# #  W zaleznosci od rozmiaru danych mozemy odpowiedznio dostosowac ilosc bitow
# c = np.array([2, 4, 6], dtype='int16')
# print(c)
# print(c.dtype)
#
# # Otrzymywanie rozmiaru macierzy
# print(a.itemsize)
# print(b.itemsize)
#
# # Otrzymywanie calkowitego rozmiaru macierzy
# print(a.nbytes)
# print(b.nbytes)

# a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
# print(a)
#
# #Otrzymywanie konkretnego elementu z macierzy
# print(a[1,5])
#
# #otrzymywanie konkretnej kolumny
# print(a[:, 2])
#
# #otrzymywanie konkretnego wiersza
# print(a[0, :])
# print(a[0, 1:6:2])

# wektor po przekątnej 5x5, wartość k zmienia wektor tworząc przy tym nowy wiersz
# f = np.diag([x for x in range(5)],k=-1)
# print(f)

# # wartość po kolumnach
# b, c = np.indices((5, 5))
# print(b[0][1])
# print(b)
# print(c)

# g=np.fromiter(range(5), dtype='int')
# print(g)

# marcin = 'Marcin'
# marcin = b'Marcin'
# mar = np.frombuffer(marcin, dtype='S6')
# print(mar)

# #podmiana istniejacej wartosci
# #np 2wiersz 3kolumna
# a[1, 2] = 20
# print(a)
#
# #inicjalizacja macierzy z samymi zerami
# b = np.zeros(5)
# print(b)
# c = np.zeros((3, 3))
# print(c)
#
# #inicjalizacja macierzy z samymi jedynkami
#
# d = np.ones(5)
# print(d)
# e = np.ones((2, 2))
# print(e)
#
# #inicjalizacja macierzy wypelniona dowolna wybrana liczba
# f = np.full((2, 2), 99)
# print(f)
#
# #inicjalizanja macierzy na podstawie innej
# g = np.full_like(a, 4)
# print(g)
#
# #macierz z losowe liczby decymalne
# h = np.random.rand(4, 2)
# print(h)
#
# #macierz z losowymi licz calkowitymi nie wiekszymi od 10
# i = np.random.randint(10, size=(3, 3))
# print(i)
#
# #macierz tozszamosciowa
# j = np.identity(5)
# print(j)
#
# #powtarzanie wartosci macierzy
# arr = np.array([[1, 2, 3]])
# r1 = np.repeat(arr, 3, axis=0)
# print(arr)
# print(r1)
#
# # przykladowa macierz na podstawie powyzszej wiedzy
#
# matrix = np.ones((5, 5))
# print(matrix)
# z = np.zeros((3, 3))
# z[1, 1] = 9
# print(z)
# matrix[1:4, 1:4] = z
# print(matrix)

############### czesc matematyczna

# a = np.array([1, 2, 3, 4])
# print('+ :', a + 2)
# print('-: ' , a - 2)
# print('*: ',a * 2)
# print('/: ',a / 2)
# print('**: ',a ** 2)
# print('sin: ',np.sin(a))
# b = np.array([4, 3, 2, 1])
# print('+: ',a + b)
# print('-:',a - b)
# print('*: ',a * b)
# print('/: ',a / b)

######### algebra liniowa


# a = np.ones((2,3))
# print(a)
#
# b = np.full((3,2),2)
# print(b)
# print('a * b: ', np.matmul(a,b))
#
# c = np.identity(3)
# print(np.linalg.det(c))

######### statystyka

# matrix = np.array([[1,2,3],[4,5,6]])
# print(matrix)
# print('min: ', np.min(matrix))
# print('max: ', np.max(matrix))
# print('max kolumna: ',np.max(matrix, axis=1))
# print('max wiersz: ',np.max(matrix, axis=0))
# print('sum kolumna: ', np.sum(matrix, axis=1))

### przeksztalcanie macierzy

# przed = np.array([[1,2,3,4],[5,6,7,8]])
# print('przed\n ', przed)

# po = przed.reshape((4,2))
# print('po\n ', po)
#
# pionowe ustawienie wektorów
#
# v1 = np.array([1,2,3,4])
# v2 = np.array([5,6,7,8])
# print(np.vstack([v1,v2,v1,v2]))
# print(np.vstack([v1,v1,v2,v2]))
#
##### stos poziomy
# h1 = np.ones((2,4))
# h2 = np.zeros((2,2))
# print(np.hstack([h1,h2]))

#ladowaniedanych z pliku txt
# plik = np.genfromtxt('data.txt', delimiter=',')
# print(plik)#
# print(plik > 5)
# print(plik[plik > 5])
# print(plik[[1,2]])

##################PANDAS#############

##tworzenie serii danych(series)
# s = pd.Series([1, 2, 5, np.nan, 6, 8])
# print(s)
# s = pd.Series([10, 12, 8, 14], index=['Ala', 'Marek',
# 'Wiesiek', 'Elonora'])
# print(s)

#Tworzenie DataFrame na podstawie slownika
# data = {'Kraj' : ['Belgia','Indie', 'Brazylia', ],
#         'Stolica' : ['Bruksela', 'New Delphi', "Brasilia"],
#         'Populacja' : [11190846, 1303171035, 207847528]}

# df = pd.DataFrame(data)
# print(df)

# Sprawdzanie typów danych w DataFrame
# print(df.dtypes)

##TWORZENIE W PROSTY SPOSÓB SERII DANYCH - czyli ptóbek
# daty = pd.date_range('20210324', periods=5)
# print(daty)
# df = pd.DataFrame(np.random.randn(5, 4), index = daty,
# columns=list('ABCD'))
# # print(df)

#uzywanie danych z zwenetrzych zrodel(CSV- odczyt zapis)
# df = pd.read_csv('dane.csv', header=0, sep=';', decimal=',')
# print(df)
# df.to_csv('plik.csv', index=False) #zapis

###EXCEL
# xlsx = pd.ExcelFile('imiona.xlsx')
# df = pd.read_excel(xlsx, header=0)
# print(df)
# df.to_excel('wyniki.xlsx', sheet_name='arkusz pierwszy')

###POBIERANIE STRUKTUR DANYCH###
# s = pd.Series([10, 12, 8, 14], index=['Ala', 'Marek', 'Wiesiek', 'Elonoora'])
# print(s)
# data = {'Kraj' : ['Belgia','Indie', 'Brazylia', ],
#        'Stolica' : ['Bruksela', 'New Delphi', "Brasilia"],
#        'Populacja' : [11190846, 1303171035, 207847528]}

# df = pd.DataFrame(data)
# print(df)

#PIJEDYNCZE ODNOSZENIE SIE DO ELEMENTU
#ZA POMOCĄ INDEKSU
# print(s['Wiesiek'])

# LUB POPRZEZ WARTOSĆ SERII
# print(s.Wiesiek)

# ...lub jak przy cieciu tablic tylko oparte na indeksach
# print(df[0:1])
# print("")

# WYSWIETLANIE KOLUMNY PO ETYKIECIE
# print(df['Populacja'])

# pobieranie pojedynczej wartosci po indeksie wiersza i kolumny
# print(df.iloc[[0], [0]])

# pobieranie wartosci po indeksie wiersza i etykiecie kolumny
# print(df.loc[[0],["Kraj"]])
# print(df.at[0,"Kraj"])

#podobnie jak w przypadku serii mozna odwolac sie do kolumn
#jak do poll klasy
#dodatkowo print jest wywolywany jak w petli dla kazdego
#elementu danej do olumny
# print('Kraj: ' + df.Kraj)



#Pandas posiada rowniez funkcje pozwalajace na losowe pobieranie elementow
#lub w odniesieniu do procentowej wielkosci calego zbioru

#jeden losowy element
# print(df.sample())

# n losowych elementow
# print(df.sample(2))

#ilosc elementow procentowo, uwaga na zaokroglenie
# print(df.sample(frac=0.5))

#jezeli potrzeba nam wiecej probek niz znajduje sie w zbiorze
# mozemy dopuscić duplikaty
# print(df.sample(n = 10, replace=True))

#zamiast wyswietlac cale kolekcje mozemy wyswietlic
# okreslona ilosc elemetow od poczatku lub konca
# print(df.head())
# print(df.head(2))
# print(df.tail(1))

##STATYSTYKA PANDAS
# print(df.describe())
# #transpozycja to zmienna T kolekcji, podobnie jak w numpy
# print(df.T)

####filtrowanie, grupowanie, agregacja danych

# s = pd.Series([10, 12, 8, 14], index=['Ala', 'Marek', 'Wiesiek', 'Elonoora'])
# # print(s)
#
# data = {'Kraj' : ['Belgia','Indie', 'Brazylia', ],
#        'Stolica' : ['Bruksela', 'New Delphi', "Brasilia"],
#        'Populacja' : [11190846, 1303171035, 207847528]}
#
# df = pd.DataFrame(data)
#
# #wyswietlanie danych serii w zaleznosci od wartosci
# print(s[s>9])

# #lub
# print(s.where(s>10))

# #mozemy rownierz podmieniac wartosci
# print(s.where(s>10, 'za duze'))

# #mozemy podmienic wartosc w oryginale(domyslnie zwracana jest kopia)
# seria = s.copy()
# seria.where(seria > 10, 'za duze', inplace=True)
# print(seria)
#
# #wyswietlanie wartosci mniejszych np. od 10

# print(s[~(s>10)])

# # mozemy rowniez loczyc warunki
# print(s[(s<13)& (s>8)])
#
### warunki dla pobieranie data frame
#   print(df[df['Populacja']>1200000000])

# # bardziej skomplikowane warunki
# print(df[(df.Populacja)>1000000 & (df.index.isin([0,2]))])

# # inny prsyklad z lista dopuszczalnych wartosci oraz isin
# #zwacajaca wartosci boolowskie
# print('#########')
# szukaj = ['Belgia','Brasilia']
# print(df.isin(szukaj))
#
# ## zmiana, usuwanie i dodawanie danych
#
# # w prypadku serii mozemy dodac/zmienic wartosc poprzez odwolanie sie
# # do elementu serii przez klucz (index)
#
# s['Wiesiek'] = 15
# print(s.Wiesiek)
# s['Alan'] = 16
# print(s)
#
# #podobna operacja dla Data frame ma nieco inny efekt- wartosc
# #ustawiona dla wszystkichh kolumn
# df.loc[3] = 'dodane'
# print(df)
# #ale mcana dodac wiersz w postaci licty
# df.loc[4] = ['Polska','Warszawa', 38675467]
# print(df)

#usuwanie danych moza wykonywac przez funkcje drop, alepamietajmy
# ze operacja nie wykonuje sie in-place wiec
#zwracana jest kopia DataFramr z usunietymi  wartosciami
# new_df = df.drop([3])
# print(new_df)

# #aby zmienic oryginal nalezy dodac inplace
# df.drop([3], inplace=True)
# print(df)

# #moza usuwac cale kolumny po nazwie indeksu ale wykonanie trj czynnosci
# #uniemozliwi dalsze wykonywanie kodu
# #df.drop('Kraj',axis=1, inplace=True)
#
# #do DataFrame mozemy dodawac rowniez kolumny zamiast wierszy
# df['Kontynent']= ['Europa', 'Azja', 'Ameryka Poludniowa', 'Europa']
# print(df)
#
# #Pandas ma rowniez wlasne funkcje sortowania danych
# print(df.sort_values(by='Kraj'))
#
# #grupowania
# grouped = df.groupby(['Kontynent'])
# print(grouped.get_group('Europa'))
#
# # mozna tez jak w SQL czy Excelu uruchomic funkcje agregujace na danej kolumnie
# print("###")
# print(df.groupby(['Kontynent']).agg({'Populacja':['sum']}))

###WYKRESY
ts = pd.Series(np.random.randn(1000))

#funkcja biblioteki pandas generujaca skumulowana sume kolejnych elementow
# ts = ts.cumsum()
# print(ts)
# ts.plot()
# plt.show()

# data = {'Kraj' : ['Belgia','Indie', 'Brazylia','Polska' ],
#         'Stolica' : ['Bruksela', 'New Delphi', "Brasilia",'Warszawa'],
#         'Kontynent':['Europa','Azja','Ameryka Poludnowa','Europa'],
#         'Populacja' : [11190846, 1303171035, 207847528,38675467]}

# df = pd.DataFrame(data)
# print(df)
# grupa = df.groupby(['Kontynent']).agg({'Populacja':['sum']})
# print(grupa)

#tworzenie wykresu
# grupa.plot(kind='bar', xlabel='Kontynent',ylabel='Mld', rot=0,
#            legend=True, title='Populacja z podzialem na kontynenty')
# plt.show()

##LUB

# wykres = grupa.plot.bar()
# wykres.set_ylabel("MLD")
# wykres.set_xlabel('Kontynent')
# wykres.tick_params(axis='x', labelrotation=0)
# wykres.legend()
# wykres.set_title('Populacja z podzialem na kontynenty')
# plt.savefig('wykres.png')
# plt.show()

###wczytywanie danych z pliku i wyswietlanie zgrupowanych wartosci
# df = pd.read_csv('dane.csv', header=0, sep=';', decimal='.')
# print(df)
# grupa = df.groupby(['Imię i nazwisko']).agg({'Wartość zamówienia':['sum']})

#wykres kolumnowy z wartosciami procentowymi sformatowanymmi z dokladnoscia
# do 2 miejsc po przecinku
#figsize ustawia wielkosc wykresu w calach,  domyslnie [6.4, 4.8]

# grupa.plot(kind='pie', subplots=True, autopct='%.2f%%', fontsize=20, figsize=(6, 6),
#            colors=['red', 'green'])
#
# plt.legend(loc="lower right")
# plt.title('Suma zamowienia dla sprzedawcy')
# plt.show()


##wykres sredniej kroczącej

# ts = ts.cumsum()
# df = pd.DataFrame(ts, columns=['wartosci'])
# #dodanie nowej kolumny i wykorzystanie funkcji rolling do stworzniea
# #kolejnych wrtosci średniej kroczącej
# df['Średnia krocząca'] = df.rolling(window=50).mean()
# df.plot()
# plt.legend()
# plt.show()

from PIL import Image
# plt.plot([1,2,3,4],[5,6,7,8])
# plt.ylabel('wartosci')
# plt.show()
#
# plt.plot([1,2,3,4],[1,4,9,16], 'r:')
# plt.plot([1,2,3,4],[1,4,9,16], 'bo')
#
# plt.axis([0,6,0,20])
# plt.show()
#
# t = np.arange(0,5, 0.1)
# plt.plot(t, t, 'r-', t, t**2, 'b:', t,t**3, 'g')
# plt.legend(labels=['liniowa', 'kwadratowa', 'szescnienna'], loc= 'center left')
# plt.show()
#
# x = np.linspace(0, 2, 100)
#
# plt.plot(x, x, label= 'liniowa')
# plt.plot(x, x**2, label= 'kwadratowa')
# plt.plot(x, x**3, label= 'szecienna')
# plt.xlabel('etykieta osi x')
# plt.ylabel('etykieta osi y')
# plt.title('Wykres trzech linii')
#
# plt.savefig('plot.png')
# plt.show()
#
# im1 = Image.open('plot.png')
# im1 = im1.convert('RGB')
# im1.save('plot1.jpg')

# x = np.arange(1, 20)
# y = 1/x
# plt.plot(x, y, 'o-')
# plt.show()

# x = np.arange(1, 20)
# y = np.sin(x)
#
# plt.plot(x, y, 'ro', label='F(x) = sinx')
# plt.legend(labels=['F(x) = sinx'])
# plt.grid()
# plt.xlabel('wartosc x')
# plt.ylabel('wartosc F(x)')
# plt.show()

# x1 = np.arange(0, 2, 0.02)
# x2 = np.arange(0, 2, 0.02)
#
# y1 = np.sin(2 * np.pi * x1)
# y2 = np.cos(2 * np.pi * x2)
# #
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1)
# plt.ylabel('sin(x)')
# plt.title('wykres sin(x)')
#
# plt.subplot(2, 1, 1)
# plt.plot(x2, y2, 'r-')
# plt.ylabel('cos(x)')
# plt.title('wykres cos(x)')
# plt.show()


# fig, axs = plt.subplots(3, 2)
#
# axs[0, 0].plot(x1, y1, 'g-')
# axs[0, 0].set_xlabel('x')
# axs[0, 0].set_ylabel('y')
# axs[0, 0].set_title('wyres sin(x)')
#
#
#
# axs[1, 1].plot(x2, y2, 'r-')
# axs[1, 1].set_xlabel('x')
# axs[1, 1].set_ylabel('cos(x)')
# axs[1, 1].set_title('Wykres cos(x)')
#
# axs[2, 0].plot(x2, y2, 'r-')
# axs[2, 0].set_xlabel('x')
# axs[2, 0].set_ylabel('cos(x)')
# axs[2, 0].set_title('Wykres cos(x)')
#
# fig.delaxes(axs[0, 1])
# fig.delaxes(axs[1, 0])
# fig.delaxes(axs[2, 1])
# plt.show()


###########WWYkresy


# data = {'a':np.arange(50),
#         'c':np.random.randint(0, 51 ,50),
#         'd':np.random.rand(50)}
#
# data['b'] = data['a']+10 * np.random.random(50)
# data['d'] = np.abs(data['d']) *100
#
# plt.scatter(data=data, x= 'a', y='b', c='c', s='d' )
# plt.xlabel('wartosci z klucza a')
# plt.ylabel('wartosci klocza b')
# #plt.show()
#
# data = {'Kraj' : ['Belgia','Indie', 'Brazylia', 'Polska' ],
#       'Stolica' : ['Bruksela', 'New Delphi', "Brasilia", 'Warszawa'],
#       'Populacja' : [11190846, 1303171035, 207847528, 20784478],
#         'Kontynent':['Europa', 'Azja', 'Ameryka Południowa', 'Europa']}
#
# df = pd.DataFrame(data)
# print(df)
# grupa = df.groupby('Kontynent')
# etykiety = list(grupa.groups.keys())
# wartosci = list(grupa.agg('Populacja').sum())
# print(etykiety)
# print(wartosci)
# plt.bar(x=etykiety, height=wartosci, color=['red','green','blue'])
# plt.xlabel('Kontynent')
# plt.ylabel('Populacja na kontynentach')
# #plt.show()
#
# x = np.random.randn(10000)
# plt.hist(x, bins=50, facecolor='g', alpha=0.75, density=True)
# plt.xlabel('Wartosci')
# plt.ylabel('Prawdopodobienstwa')
# plt.title('Histogram')
# plt.show()
#
#
#
