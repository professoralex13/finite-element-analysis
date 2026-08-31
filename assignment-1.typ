#let labelmat(
  collabels,
  rowlabels,
  ..args,
) = context {
  let numcols = collabels.len()
  let numrows = rowlabels.len()
  let matentries = args.pos().chunks(numcols)
  let matheight = (
    matentries
      .map(
        row => calc.max(..row.map(i => measure(i).height)),
      )
      .sum()
      + 10pt * numrows
  )
  let delimcell(delim) = table.cell(
    rowspan: numrows,
    box(inset: (top: -5pt, left: -5pt), $lr(delim, size: #matheight)$),
  )
  table(
    columns: (auto, 7pt, ..(auto,) * numcols, 7pt),
    stroke: none,
    ..args.named(),
    [], [], ..collabels, [],
    ..for (rowindx, (rowlab, rowentries)) in rowlabels.zip(matentries).enumerate() {
      (
        rowlab,
        ..if rowindx == 0 { (delimcell($\[$),) },
        ..rowentries,
        ..if rowindx == 0 { (delimcell($\]$),) },
      )
    },
  )
}

#set page(paper: "a4", margin: 1cm)

#align(center)[
  #text(size: 20pt)[
    ENME302 \
    Assignment 1 \
    Alex Cutforth \
    24375019 \
    3rd year Mechatronics
  ]
]

= Introduction
= Methods
This system is modelled using 7 frame elements, with all joints between them being weldsW, meaning only 1 rotational degree of freedom.
The only two nodes which do not have 3 degrees of freedom are the left supports, which as fixed joints, have none.
This method of modeling relies on the assumption of small deflections, where $sin(theta) approx theta$ and $cos(theta) approx 1$. This allows the governing equations to use linear approximations.

= Results/Discussion

In part 1, the wind on the sign is modelled using two 2kN forces in the X axis at the top and bottom of the sign elements. This results in a deflection at the top of the sign of: $x=-3.02"mm",y=1.06"mm","rot"=3.44"mrad"$. This means the tip of the sign is rotating up and backward, which makes sense given the wind direction and location of fixed supports.

= Conclusion


#set page(columns: 2)

= Appendix
#show figure.where(kind: table): set figure.caption(position: top)

#let local_dofs = ($D_1$, $D_2$, $D_3$, $D_4$, $D_5$, $D_6$)
#let unit_row6(pos) = range(6).map(i => if i == pos { $1$ } else { $0$ })

#let assembly1_rows = (
  unit_row6(3),
  unit_row6(4),
  unit_row6(5),
  ($dots.v$,) * 6,
  ($0$,) * 6,
)

#figure(
  labelmat(
    local_dofs,
    ($q_1$, $q_2$, $q_3$, $dots.v$, $q_15$),
    ..assembly1_rows.flatten(),
    align: center + horizon,
  ),
  kind: table,
  caption: [Element 1 Assembly Matrix],
)

#let assembly2_rows = (
  unit_row6(3),
  unit_row6(4),
  unit_row6(5),
  ($dots.v$,) * 6,
  ($0$,) * 6,
)

#figure(
  labelmat(
    local_dofs,
    ($q_1$, $q_2$, $q_3$, $dots.v$, $q_15$),
    ..assembly2_rows.flatten(),
    align: center + horizon,
  ),
  kind: table,
  caption: [Element 2 Assembly Matrix],
)

#let assembly3_rows = (
  unit_row6(0),
  unit_row6(1),
  unit_row6(2),
  unit_row6(3),
  unit_row6(4),
  unit_row6(5),
  ($dots.v$,) * 6,
  ($0$,) * 6,
)

#figure(
  labelmat(
    local_dofs,
    ($q_1$, $q_2$, $q_3$, $q_4$, $q_5$, $q_6$, $dots.v$, $q_15$),
    ..assembly3_rows.flatten(),
    align: center + horizon,
  ),
  kind: table,
  caption: [Element 3 Assembly Matrix],
)


#let assembly4_rows = (
  ($0$,) * 6,
  ($dots.v$,) * 6,
  unit_row6(0),
  unit_row6(1),
  unit_row6(2),
  unit_row6(3),
  unit_row6(4),
  unit_row6(5),
  ($dots.v$,) * 6,
  ($0$,) * 6,
)

#figure(
  labelmat(
    local_dofs,
    ($q_1$, $dots.v$, $q_4$, $q_5$, $q_6$, $q_7$, $q_8$, $q_9$, $dots.v$, $q_15$),
    ..assembly4_rows.flatten(),
    align: center + horizon,
  ),
  kind: table,
  caption: [Element 4 Assembly Matrix],
)

#colbreak()


#let assembly5_rows = (
  ($0$,) * 6,
  ($dots.v$,) * 6,
  unit_row6(0),
  unit_row6(1),
  unit_row6(2),
  ($dots.v$,) * 6,
  unit_row6(3),
  unit_row6(4),
  unit_row6(5),
  ($dots.v$,) * 6,
  ($0$,) * 6,
)

#figure(
  labelmat(
    local_dofs,
    ($q_1$, $dots.v$, $q_4$, $q_5$, $q_6$, $dots.v$, $q_10$, $q_11$, $q_12$, $dots.v$, $q_15$),
    ..assembly5_rows.flatten(),
    align: center + horizon,
  ),
  kind: table,
  caption: [Element 5 Assembly Matrix],
)


#let assembly6_rows = (
  ($0$,) * 6,
  ($dots.v$,) * 6,
  unit_row6(0),
  unit_row6(1),
  unit_row6(2),
  unit_row6(3),
  unit_row6(4),
  unit_row6(5),
  ($dots.v$,) * 6,
  ($0$,) * 6,
)

#figure(
  labelmat(
    local_dofs,
    ($q_1$, $dots.v$, $q_7$, $q_8$, $q_9$, $q_10$, $q_11$, $q_12$, $dots.v$, $q_15$),
    ..assembly6_rows.flatten(),
    align: center + horizon,
  ),
  kind: table,
  caption: [Element 6 Assembly Matrix],
)


#let assembly7_rows = (
  ($0$,) * 6,
  ($dots.v$,) * 6,
  unit_row6(0),
  unit_row6(1),
  unit_row6(2),
  unit_row6(3),
  unit_row6(4),
  unit_row6(5),
)

#figure(
  labelmat(
    local_dofs,
    ($q_1$, $dots.v$, $q_10$, $q_11$, $q_12$, $q_13$, $q_14$, $q_15$),
    ..assembly7_rows.flatten(),
    align: center + horizon,
  ),
  kind: table,
  caption: [Element 7 Assembly Matrix],
)
