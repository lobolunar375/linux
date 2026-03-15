Level 0 -> Level 1
Objetivo: Conectarse al servidor vía SSH.

Comando: ssh bandit0@bandit.labs.overthewire.org -p 2220

Password: bandit0

Level 1 -> Level 2
Objetivo: Leer un archivo llamado - en el directorio home.

Nota: El guion - es un carácter especial en Linux. Para leerlo, se debe especificar la ruta.

Comando: cat ./-

Level 2 -> Level 3
Objetivo: Leer un archivo con espacios en el nombre: spaces in this filename.

Comando: cat "spaces in this filename" o cat spaces\ in\ this\ filename

Level 3 -> Level 4
Objetivo: Leer un archivo oculto dentro del directorio inhere.

Comando: ls -la inhere/ -> cat inhere/.hidden

Level 4 -> Level 5
Objetivo: Encontrar el único archivo legible por humanos en el directorio inhere.

Comando: file inhere/* -> cat inhere/-file07

Level 5 -> Level 6
Objetivo: Buscar un archivo con propiedades específicas: 1033 bytes, no ejecutable, legible.

Comando: find inhere/ -type f -size 1033c ! -executable

Level 6 -> Level 7
Objetivo: Buscar un archivo en todo el servidor basado en el usuario y grupo.

Comando: find / -user bandit7 -group bandit6 -size 33c 2>/dev/null

Level 7 -> Level 8
Objetivo: Extraer la contraseña al lado de la palabra "millionth" en data.txt.

Comando: grep "millionth" data.txt

Level 8 -> Level 9
Objetivo: Encontrar la única línea de texto que no se repite.

Comando: sort data.txt | uniq -u

Level 9 -> Level 10
Objetivo: Extraer texto legible de un archivo binario precedido por caracteres =.

Comando: strings data.txt | grep "=="

Level 10 -> Level 11
Objetivo: Decodificar un archivo en formato Base64.

Comando: base64 -d data.txt

Level 11 -> Level 12
Objetivo: Descifrar un texto rotado 13 posiciones (ROT13).

Comando: cat data.txt | tr 'A-Za-z' 'N-ZA-Mn-za-m'

Level 12 -> Level 13
Objetivo: Descomprimir repetidamente un archivo (gzip, bzip2, tar) extraído de un hex dump.

Comandos clave:

xxd -r data.txt > data_bin

file data_bin (para identificar el tipo).

gunzip, bunzip2, tar -xf según corresponda.

Level 13 -> Level 14
Objetivo: Usar una clave SSH privada para conectarse como bandit14.

Comando: ssh -i sshkey.private bandit14@localhost -p 2220

