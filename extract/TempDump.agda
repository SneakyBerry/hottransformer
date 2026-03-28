{-# OPTIONS --guardedness --erased-cubical #-}
module TempDump where

open import Data.List.Relation.Unary.Any.Properties
open import Extractor
open import DumpModule

test = dump lookup-result
