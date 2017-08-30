test.raw
table(complete.cases(test.raw))
map_int(test.raw,~sum(is.na(.x)))
