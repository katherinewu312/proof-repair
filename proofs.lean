import Init.Data.List.Basic
import Init.Data.Char.Basic
import Init.Data.Option.Basic
import Init.Data.String.Basic

open String

theorem theorem_38179 (s : String) (i : Pos.Raw) (h : i ≠ 0) :
    (i.prev s).1 < i.1 := by
  exact Pos.Raw.prev_lt_of_pos s i h
