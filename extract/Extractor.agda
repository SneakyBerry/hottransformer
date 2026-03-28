{-# OPTIONS --guardedness --erased-cubical #-}
module Extractor where

open import Agda.Builtin.Reflection
open import Agda.Builtin.List
open import Agda.Builtin.Nat
open import Agda.Builtin.String
open import Agda.Builtin.Bool
open import Agda.Builtin.Char
open import Agda.Builtin.Unit renaming (⊤ to Unit; tt to unit)

-- String building helpers
infixr 5 _++_
_++_ : String → String → String
_++_ = primStringAppend

showNat : Nat → String
showNat zero = "0"
showNat (suc zero) = "1"
showNat (suc (suc zero)) = "2"
showNat (suc (suc (suc zero))) = "3"
showNat (suc (suc (suc (suc zero)))) = "4"
showNat (suc (suc (suc (suc (suc zero))))) = "5"
showNat (suc (suc (suc (suc (suc (suc zero)))))) = "6"
showNat (suc (suc (suc (suc (suc (suc (suc zero))))))) = "7"
showNat (suc (suc (suc (suc (suc (suc (suc (suc zero)))))))) = "8"
showNat (suc (suc (suc (suc (suc (suc (suc (suc (suc zero))))))))) = "9"
showNat (suc (suc (suc (suc (suc (suc (suc (suc (suc (suc n)))))))))) = "10"

-- JSON helpers
jsonStr : String → String
jsonStr s = "\"" ++ s ++ "\""

jsonField : String → String → String
jsonField key val = jsonStr key ++ ":" ++ val

jsonObj : List (String) → String
jsonObj [] = "{}"
jsonObj (x ∷ []) = "{" ++ x ++ "}"
jsonObj (x ∷ xs) = "{" ++ x ++ "," ++ jsonObjTail xs
  where
  jsonObjTail : List String → String
  jsonObjTail [] = "}"
  jsonObjTail (y ∷ []) = y ++ "}"
  jsonObjTail (y ∷ ys) = y ++ "," ++ jsonObjTail ys

jsonArr : List String → String
jsonArr [] = "[]"
jsonArr (x ∷ []) = "[" ++ x ++ "]"
jsonArr (x ∷ xs) = "[" ++ x ++ "," ++ jsonArrTail xs
  where
  jsonArrTail : List String → String
  jsonArrTail [] = "]"
  jsonArrTail (y ∷ []) = y ++ "]"
  jsonArrTail (y ∷ ys) = y ++ "," ++ jsonArrTail ys

-- Serialize Visibility
showVis : Visibility → String
showVis visible = jsonStr "visible"
showVis hidden = jsonStr "hidden"
showVis instance′ = jsonStr "instance"

-- Serialize reflected Term to JSON
{-# TERMINATING #-}
showTerm : Term → String
showArg : Arg Term → String
showArgs : List (Arg Term) → String
showAbs : Abs Term → String
showSort : Sort → String
showClause : Clause → String
showClauses : List Clause → String
showPattern : Pattern → String
showPatterns : List (Arg Pattern) → String

showTerm (var x args) =
  jsonObj (jsonField "tag" (jsonStr "var")
         ∷ jsonField "index" (showNat x)
         ∷ jsonField "args" (showArgs args)
         ∷ [])
showTerm (con c args) =
  jsonObj (jsonField "tag" (jsonStr "con")
         ∷ jsonField "name" (jsonStr (primShowQName c))
         ∷ jsonField "args" (showArgs args)
         ∷ [])
showTerm (def f args) =
  jsonObj (jsonField "tag" (jsonStr "def")
         ∷ jsonField "name" (jsonStr (primShowQName f))
         ∷ jsonField "args" (showArgs args)
         ∷ [])
showTerm (lam v t) =
  jsonObj (jsonField "tag" (jsonStr "lam")
         ∷ jsonField "visibility" (showVis v)
         ∷ jsonField "body" (showAbs t)
         ∷ [])
showTerm (pat-lam cs args) =
  jsonObj (jsonField "tag" (jsonStr "pat-lam")
         ∷ jsonField "clauses" (showClauses cs)
         ∷ jsonField "args" (showArgs args)
         ∷ [])
showTerm (pi a b) =
  jsonObj (jsonField "tag" (jsonStr "pi")
         ∷ jsonField "domain" (showArg a)
         ∷ jsonField "codomain" (showAbs b)
         ∷ [])
showTerm (agda-sort s) =
  jsonObj (jsonField "tag" (jsonStr "sort")
         ∷ jsonField "sort" (showSort s)
         ∷ [])
showTerm (lit l) =
  jsonObj (jsonField "tag" (jsonStr "lit")
         ∷ jsonField "value" (jsonStr (primShowChar 'L'))
         ∷ [])
showTerm (meta x args) =
  jsonObj (jsonField "tag" (jsonStr "meta")
         ∷ jsonField "args" (showArgs args)
         ∷ [])
showTerm unknown =
  jsonObj (jsonField "tag" (jsonStr "unknown") ∷ [])

showArg (arg (arg-info v (modality relevant quantity-ω)) t) =
  jsonObj (jsonField "visibility" (showVis v)
         ∷ jsonField "term" (showTerm t)
         ∷ [])
showArg (arg _ t) =
  jsonObj (jsonField "visibility" (jsonStr "other")
         ∷ jsonField "term" (showTerm t)
         ∷ [])

showArgs [] = "[]"
showArgs (x ∷ []) = "[" ++ showArg x ++ "]"
showArgs (x ∷ xs) = "[" ++ showArg x ++ "," ++ showArgsTail xs
  where
  showArgsTail : List (Arg Term) → String
  showArgsTail [] = "]"
  showArgsTail (y ∷ []) = showArg y ++ "]"
  showArgsTail (y ∷ ys) = showArg y ++ "," ++ showArgsTail ys

showAbs (abs s t) = jsonObj (jsonField "name" (jsonStr s)
                           ∷ jsonField "body" (showTerm t)
                           ∷ [])

showSort (set t) = jsonObj (jsonField "kind" (jsonStr "set") ∷ jsonField "level" (showTerm t) ∷ [])
showSort (lit n) = jsonObj (jsonField "kind" (jsonStr "set-lit") ∷ jsonField "level" (showNat n) ∷ [])
showSort (prop t) = jsonObj (jsonField "kind" (jsonStr "prop") ∷ jsonField "level" (showTerm t) ∷ [])
showSort (propLit n) = jsonObj (jsonField "kind" (jsonStr "prop-lit") ∷ jsonField "level" (showNat n) ∷ [])
showSort (inf n) = jsonObj (jsonField "kind" (jsonStr "inf") ∷ jsonField "level" (showNat n) ∷ [])
showSort unknown = jsonObj (jsonField "kind" (jsonStr "unknown") ∷ [])

showClause (clause tel ps t) =
  jsonObj (jsonField "patterns" (showPatterns ps)
         ∷ jsonField "body" (showTerm t)
         ∷ [])
showClause (absurd-clause tel ps) =
  jsonObj (jsonField "patterns" (showPatterns ps)
         ∷ jsonField "absurd" (jsonStr "true")
         ∷ [])

showClauses [] = "[]"
showClauses (x ∷ xs) = jsonArr (showClause x ∷ map showClause xs)
  where
  map : {A B : _} → (A → B) → List A → List B
  map f [] = []
  map f (y ∷ ys) = f y ∷ map f ys

showPattern (con c ps) = jsonObj (jsonField "tag" (jsonStr "pcon") ∷ jsonField "name" (jsonStr (primShowQName c)) ∷ [])
showPattern (dot t) = jsonObj (jsonField "tag" (jsonStr "pdot") ∷ [])
showPattern (var x) = jsonObj (jsonField "tag" (jsonStr "pvar") ∷ jsonField "index" (showNat x) ∷ [])
showPattern (lit l) = jsonObj (jsonField "tag" (jsonStr "plit") ∷ [])
showPattern (proj f) = jsonObj (jsonField "tag" (jsonStr "pproj") ∷ jsonField "name" (jsonStr (primShowQName f)) ∷ [])
showPattern (absurd x) = jsonObj (jsonField "tag" (jsonStr "pabsurd") ∷ [])

showPatterns [] = "[]"
showPatterns (arg _ p ∷ []) = "[" ++ showPattern p ++ "]"
showPatterns (arg _ p ∷ ps) = "[" ++ showPattern p ++ "," ++ showPatternsTail ps
  where
  showPatternsTail : List (Arg Pattern) → String
  showPatternsTail [] = "]"
  showPatternsTail (arg _ q ∷ []) = showPattern q ++ "]"
  showPatternsTail (arg _ q ∷ qs) = showPattern q ++ "," ++ showPatternsTail qs
