import numpy as np
import pandas as pd
import matplotlib as plt
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

# przeksztalcanie macierzy

# przed = np.array([[1,2,3,4],[5,6,7,8]])
# print('przed\n ', przed)
# po = przed.reshape((4,2))
# print('po\n ', po)
#
# #wektory ukladajace sie pionowo
#
# v1 = np.array([1,2,3,4])
# v2 = np.array([5,6,7,8])
#
# print(np.vstack([v1,v2,v1,v2]))
# print(np.vstack([v1,v1,v2,v2]))
#
# #stos poziomy
# h1 = np.ones((2,4))
# h2 = np.zeros((2,2))
# print(np.hstack([h1,h2]))

#ladowaniedanych z pliku txt
# plik = np.genfromtxt('data.txt', delimiter=',')
# print(plik)
#
# #############33 zaawansowane indeksowanie i boolean masking
#
# print(plik > 5)
# print(plik[plik > 5])
# print(plik[[1,2]])

##################PANDAS#############
#tworzenie serii danych(series)

# s = pd.Series([1, 2, 5, np.nan, 6, 8])
# print(s)
# s = pd.Series([10, 12, 8, 14], index=['Ala', 'Marek',
# 'Wiesiek', 'Elonora'])
# print(s)

#Tworzenie DataFrame na podstawie slownika

# data = {'Kraj' : ['Belgia','Indie', 'Brazylia', ],
#         'Stolica' : ['Bruksela', 'New Delphi', "Brasilia"],
#         'Populacja' : [11190846, 1303171035, 207847528]}
#
# df = pd.DataFrame(data)
# # print(df)
# #Dataframe przechowuje typ dla kazdej kolumny co mozemy
# #sprawdzic wypisując:
# print(df.dtypes)
#
# #tworzenie w prosty sposob serii danych - czyli probki dlakolejnych
# daty = pd.date_range('20210324', periods=5)
# # print(daty)
# df = pd.DataFrame(np.random.randn(5, 4), index = daty,
# columns=list('ABCD'))
# # print(df)

#uzywanie danych z zwenetrzych zrodel(CSV- odczyt zapis)
# df = pd.read_csv('dane.csv', header=0, sep=';', decimal=',')
# print(df)
# df.to_csv('plik.csv', index=False)
#
# ##EXCEL
#
# xlsx = pd.ExcelFile('imiona.xlsx')
# df = pd.read_excel(xlsx, header=0)
# print(df)
# df.to_excel('wyniki.xlsx', sheet_name='arkusz pierwszy')

###POBIERANIE STRUKTUR DANYCH###
# s = pd.Series([10, 12, 8, 14], index=['Ala', 'Marek', 'Wiesiek', 'Elonoora'])
# print(s)
#
# data = {'Kraj' : ['Belgia','Indie', 'Brazylia', ],
#        'Stolica' : ['Bruksela', 'New Delphi', "Brasilia"],
#        'Populacja' : [11190846, 1303171035, 207847528]}
#
# df = pd.DataFrame(data)
# print(df)
#
# #pojedyncze odnoszenie sie do elementu
# #za pomoca indeksu
# print(s['Wiesiek'])
# #Lub poprzez wartosc serii jak do pola klasy
# print(s.Wiesiek)
# #...lub jak przy cieciu tablic tylko oparte na indeksach
# print(df[0:1])
# print("")
# #kolumny po  etykiecie
# print(df['Populacja'])
#
# #pobieranie pojedynczej wartosci po indeksie wiersza i kolumny
# print(df.iloc[[0], [0]])
#
# #pobieranie wartosci po indeksie wiersza i etykiecie kolumny
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
# mozemy dopuscid duplikaty
# print(df.sample(n = 10, replace=True))

#zamiast wyswietlac calp kolekcje mozemy wyswietlic
# okreslona ilosc elemetow od poczatku lub konca
# print(df.head())
# print(df.head(2))
# print(df.tail(1))

##STATYSTYKA PANDAS
# print(df.describe())
# #transpozycja to zmienna T kolekcji, podobnie jak w numpy
# print(df.T)

####filtrowanie, grupowanie, agregacja danych

s = pd.Series([10, 12, 8, 14], index=['Ala', 'Marek', 'Wiesiek', 'Elonoora'])
# print(s)

data = {'Kraj' : ['Belgia','Indie', 'Brazylia', ],
       'Stolica' : ['Bruksela', 'New Delphi', "Brasilia"],
       'Populacja' : [11190846, 1303171035, 207847528]}

df = pd.DataFrame(data)

#wyswietlanie danych serii w zaleznosci od wartosci
print(s[s>9])


















































