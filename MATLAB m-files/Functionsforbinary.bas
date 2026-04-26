Attribute VB_Name = "Functionsforbinary"
Function errorout(number, ident1, ident2, ident3, ident4)
Select Case number
Case 1
q = MsgBox("Race data where there shouldn't be" & " format " & ident1 & " Text " & ident2 & " Line " & ident3)
Case 2
q = MsgBox("Dog data where there shouldn't be" & " Line " & ident1 & " Text " & ident2 & " formatrow " & ident3)
Case 3
q = MsgBox("Data length is zero but field is not blank Line " & ident1 & " Text " & ident2 & " format counnter " & ident3 & " race " & ident4)

End Select
End Function

Function translatetext(formattype, formatparam, formatparam2, carrystring)

Select Case formattype
    Case "BLANK"    'shouldn't be needed as this function is skipped if format is blank
    binout = ""
    Case "LAST NUMBERS"
    Dummy1 = Right(carrystring, CInt(formatparam))
    binout = binaryof(Dummy1, formatparam2)
    Case "MAX"
    binout = binaryof(carrystring, CInt(formatparam2))
    Case "LAST LETTERS"
    binout = ""
        For z = 1 To formatparam
        onechar = Mid(carrystring, z, 1)
        binout = binout & binaryof(CStr(Asc(onechar) - 96), 5)
        Next z
    Case "BIT FLAGS"
    If carrystring = "0" Then carrystring = "3"
    binout = String(CInt(carrystring) - 1, "0") & "1" & String(CInt(formatparam) - CInt(carrystring), "0")
End Select

    


translatetext = binout
End Function
Function binaryof(inputstringx, formatparam2)
'returns binary of a number as a text string of 1s and 0s given a maximum lengthof
inputstringnum = CLng(inputstringx)
ohones = ""

For i = 1 To CInt(formatparam2)
If inputstringnum >= 2 ^ (formatparam2 - i) Then ohones = ohones & "1": inputstringnum = inputstringnum - (2 ^ (formatparam2 - i)) Else ohones = ohones & "0"
Next i

binaryof = ohones
End Function
Function lettercodes()

numberx = InputBox("How many codes?")
lettersx = InputBox("How many letters?")
columnx = InputBox("What column?")
numbern = CInt(numberx)
lettersn = CInt(lettersx)
columnn = CInt(columnx)

For i = 1 To numberx
    ank = ""
    For x = 1 To lettersx
    ank = ank & Chr(CInt(Rnd(1) * 25) + 97)



    Next x
    Sheets("Sheet1").Cells(i, columnn).Formula = ank
Next i

End Function
