{-# OPTIONS --guardedness --erased-cubical #-}
module DumpModule where

open import Agda.Primitive using (Level; lzero; lsuc; Set)
open import Agda.Builtin.Reflection renaming (returnTC to return; Type to RType)
open import Agda.Builtin.List
open import Agda.Builtin.Nat
open import Agda.Builtin.String
open import Agda.Builtin.Unit renaming (⊤ to Unit; tt to unit)
open import Agda.Builtin.Sigma
open import Extractor

infixl 1 _>>=_
_>>=_ : ∀ {a b} {A : Set a} {B : Set b} → TC A → (A → TC B) → TC B
_>>=_ = bindTC

-- Non-macro version: call from unquoteDecl or runTC
dumpName : Name → TC Unit
dumpName n = do
  ty ← inferType (def n [])
  defn ← getDefinition n
  let tyJson = showTerm ty
  let bodyJson = showDefn defn
  let result = "{" ++ jsonField "name" (jsonStr (primShowQName n))
                   ++ "," ++ jsonField "type" tyJson
                   ++ "," ++ jsonField "body" bodyJson
                   ++ "}"
  typeError (strErr result ∷ [])
  where
  showDefn : Definition → String
  showDefn (function cs) = showClauses cs
  showDefn (data-type pars cs) =
    jsonObj (jsonField "kind" (jsonStr "data")
           ∷ jsonField "pars" (showNat pars)
           ∷ [])
  showDefn (record-type c fs) =
    jsonObj (jsonField "kind" (jsonStr "record")
           ∷ jsonField "con" (jsonStr (primShowQName c))
           ∷ [])
  showDefn (data-cons d q) =
    jsonObj (jsonField "kind" (jsonStr "data-con")
           ∷ jsonField "data" (jsonStr (primShowQName d))
           ∷ [])
  showDefn axiom =
    jsonObj (jsonField "kind" (jsonStr "axiom") ∷ [])
  showDefn prim-fun =
    jsonObj (jsonField "kind" (jsonStr "prim") ∷ [])

-- Macro wrapper: use as `dump (quote name)`
macro
  dump : Name → Term → TC Unit
  dump n _ = dumpName n
