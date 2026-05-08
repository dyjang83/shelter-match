"""
shelter.py
==========

Python port of the C++ midterm shelter design. Preserves the original
object-oriented structure:

  - Animal is an abstract base class (ABC + abstractmethod) — same role as
    the pure-virtual Animal in the C++ version.
  - Dog, Cat, Rabbit override info() polymorphically.
  - AdoptionRecord forms a recursive linked chain of past adopters; the
    print_history() method walks it recursively, mirroring the C++ design.
  - Each animal carries ML-relevant traits (energy, size, etc.) that the
    matching model uses as features.

Design note: in C++ we used unique_ptr to own the recursive AdoptionRecord
chain. Python's reference counting + garbage collection handles the same job
for free, so the chain is just a regular attribute that points to the
previous record (or None at the base case).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import uuid


# -----------------------------------------------------------------------------
# AdoptionRecord — recursive chain, mirrors the C++ struct
# -----------------------------------------------------------------------------
@dataclass
class AdoptionRecord:
    """A node in the recursive adoption-history chain.

    Each record stores one adopter's name and a link to the previous record
    (the older adoption). The chain reads most-recent -> oldest.
    """
    adopter_name: str
    previous: Optional["AdoptionRecord"] = None

    def print_history(self, indent: int = 0) -> None:
        """Recursively print the full adoption chain, newest first.

        Base case: previous is None (no earlier adoption).
        Recursive step: delegate to previous.print_history().
        """
        prefix = "  " * indent
        print(f"{prefix}{self.adopter_name}")
        if self.previous is not None:
            print(f"{prefix}  Previously adopted by:")
            self.previous.print_history(indent + 1)

    def to_list(self) -> list[str]:
        """Return the chain as a flat list (newest -> oldest).

        Useful for the web UI, which renders the chain as a visual timeline
        rather than printing it.
        """
        names = [self.adopter_name]
        if self.previous is not None:
            names.extend(self.previous.to_list())
        return names

    def depth(self) -> int:
        """Recursive depth of the chain (number of past adoptions)."""
        return 1 + (self.previous.depth() if self.previous else 0)


# -----------------------------------------------------------------------------
# Animal — abstract base class
# -----------------------------------------------------------------------------
class Animal(ABC):
    """Abstract base class for all shelter animals.

    Carries shared identity (name, age) plus ML feature attributes that the
    matching model consumes. Concrete subclasses (Dog, Cat, Rabbit) override
    `info()` and supply species-specific defaults via class attributes.
    """

    species: str = "animal"  # overridden by subclasses

    def __init__(
        self,
        name: str,
        age: int,
        energy_level: int = 5,        # 1 (couch potato) - 10 (zoomies)
        size: str = "medium",         # "small", "medium", "large"
        good_with_kids: bool = True,
        needs_quiet: bool = False,
        training_level: int = 5,      # 1 (untrained) - 10 (well trained)
    ):
        self._id = str(uuid.uuid4())[:8]  # short stable id for the UI
        self._name = name
        self._age = age
        self._energy_level = energy_level
        self._size = size
        self._good_with_kids = good_with_kids
        self._needs_quiet = needs_quiet
        self._training_level = training_level
        self._history: Optional[AdoptionRecord] = None

    # ---- Encapsulated accessors (mirrors the C++ getters) ----
    @property
    def id(self) -> str: return self._id
    @property
    def name(self) -> str: return self._name
    @property
    def age(self) -> int: return self._age
    @property
    def energy_level(self) -> int: return self._energy_level
    @property
    def size(self) -> str: return self._size
    @property
    def good_with_kids(self) -> bool: return self._good_with_kids
    @property
    def needs_quiet(self) -> bool: return self._needs_quiet
    @property
    def training_level(self) -> int: return self._training_level
    @property
    def history(self) -> Optional[AdoptionRecord]: return self._history

    # ---- Polymorphic interface ----
    @abstractmethod
    def info(self) -> str:
        """Return a one-line description. Each subclass overrides this."""
        ...

    # ---- Adoption mechanics (mirrors C++ recordAdoption) ----
    def record_adoption(self, adopter_name: str) -> None:
        """Push a new adoption record onto the front of the history chain.

        The new record's `previous` field points to the existing chain, then
        the new record becomes the head. Same shape as the C++ unique_ptr
        version, just without the manual move() calls.
        """
        new_record = AdoptionRecord(
            adopter_name=adopter_name,
            previous=self._history,
        )
        self._history = new_record

    def has_history(self) -> bool:
        return self._history is not None

    # ---- ML feature vector ----
    def feature_dict(self) -> dict:
        """Return this animal's features as a dict for the ML model.

        Keeping this on the Animal class (rather than a separate function)
        means each subclass could in principle override it to add
        species-specific features. For now the shared schema is enough.
        """
        return {
            "species": self.species,
            "age": self._age,
            "energy_level": self._energy_level,
            "size": self._size,
            "good_with_kids": int(self._good_with_kids),
            "needs_quiet": int(self._needs_quiet),
            "training_level": self._training_level,
        }


# -----------------------------------------------------------------------------
# Concrete animal types — polymorphic info() overrides
# -----------------------------------------------------------------------------
class Dog(Animal):
    species = "dog"

    def info(self) -> str:
        return f"{self.name} the dog (age {self.age}, energy {self.energy_level}/10)"


class Cat(Animal):
    species = "cat"

    def info(self) -> str:
        return f"{self.name} the cat (age {self.age}, energy {self.energy_level}/10)"


class Rabbit(Animal):
    species = "rabbit"

    def info(self) -> str:
        return f"{self.name} the rabbit (age {self.age}, energy {self.energy_level}/10)"


# Factory used by the UI when the user picks a species from a dropdown.
SPECIES_TO_CLASS = {
    "dog": Dog,
    "cat": Cat,
    "rabbit": Rabbit,
}


def make_animal(species: str, **kwargs) -> Animal:
    """Build an animal of the right concrete type from a species string."""
    species = species.lower().strip()
    if species not in SPECIES_TO_CLASS:
        raise ValueError(
            f"Unknown species '{species}'. Must be one of {list(SPECIES_TO_CLASS)}."
        )
    return SPECIES_TO_CLASS[species](**kwargs)


# -----------------------------------------------------------------------------
# Adopter — feature-rich profile for the ML model
# -----------------------------------------------------------------------------
@dataclass
class Adopter:
    """An adopter's lifestyle profile. Used as the second half of the
    (animal, adopter) feature pair the matching model consumes.
    """
    name: str
    lives_in_apartment: bool = False
    has_kids: bool = False
    hours_home_per_day: int = 8           # 0-24
    activity_level: int = 5               # 1 (sedentary) - 10 (very active)
    experience_level: int = 5             # 1 (first-time) - 10 (lifetime owner)
    allergic_to_cats: bool = False
    allergic_to_dogs: bool = False
    prefers_quiet_home: bool = False

    def feature_dict(self) -> dict:
        return {
            "lives_in_apartment": int(self.lives_in_apartment),
            "has_kids": int(self.has_kids),
            "hours_home_per_day": self.hours_home_per_day,
            "activity_level": self.activity_level,
            "experience_level": self.experience_level,
            "allergic_to_cats": int(self.allergic_to_cats),
            "allergic_to_dogs": int(self.allergic_to_dogs),
            "prefers_quiet_home": int(self.prefers_quiet_home),
        }


# -----------------------------------------------------------------------------
# Shelter — manages the live collection of animals
# -----------------------------------------------------------------------------
class Shelter:
    """Container for animals currently in residence. Mirrors the C++
    `vector<shared_ptr<Animal>>` plus an undo buffer for adoptions.
    """

    def __init__(self):
        self._residents: list[Animal] = []
        self._adopted: list[Animal] = []  # animals that have left (for history lookup + undo)

    @property
    def residents(self) -> list[Animal]:
        return list(self._residents)  # defensive copy

    @property
    def adopted(self) -> list[Animal]:
        return list(self._adopted)

    def intake(self, animal: Animal) -> str:
        """Add an animal to the shelter.

        If an animal with this name was previously adopted out, pull that
        same object back in so its history is preserved (matches the C++
        readmission behavior in the sample output).
        """
        for i, prev in enumerate(self._adopted):
            if prev.name == animal.name:
                returning = self._adopted.pop(i)
                self._residents.append(returning)
                return f"{returning.name} has been readmitted to the shelter."
        self._residents.append(animal)
        return f"{animal.name} the {animal.species} has been added to the shelter."

    def adopt(self, animal_name: str, adopter_name: str) -> str:
        """Adopt an animal by name. Updates history, moves animal out of
        residents and into the adopted list.
        """
        for i, animal in enumerate(self._residents):
            if animal.name == animal_name:
                animal.record_adoption(adopter_name)
                adopted = self._residents.pop(i)
                self._adopted.append(adopted)
                return f"{adopted.name} has been adopted by {adopter_name}."
        return f'No animal named "{animal_name}" is currently in the shelter.'

    def find(self, animal_name: str) -> Optional[Animal]:
        """Look up an animal by name in residents or adopted list."""
        for animal in self._residents + self._adopted:
            if animal.name == animal_name:
                return animal
        return None

    def undo_last_adoption(self) -> str:
        """Pop the most recently adopted animal back into residents."""
        if not self._adopted:
            return "Nothing to undo - no recent adoptions."
        restored = self._adopted.pop()
        self._residents.append(restored)
        return f"Undid the last adoption. {restored.name} is back in the shelter."


# -----------------------------------------------------------------------------
# Quick smoke test when run directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    s = Shelter()
    print(s.intake(Dog("Buddy", 4, energy_level=8, size="large", good_with_kids=True)))
    print(s.intake(Cat("Whiskers", 2, energy_level=3, needs_quiet=True)))
    print(s.intake(Rabbit("Mochi", 1, energy_level=4, needs_quiet=True, size="small")))

    print(s.adopt("Buddy", "Alice"))
    print(s.intake(Dog("Buddy", 4, energy_level=8, size="large")))  # readmission
    print(s.adopt("Buddy", "Carla"))

    buddy = s.find("Buddy")
    print(f"\nAdoption history for {buddy.name}:")
    if buddy.history:
        buddy.history.print_history()
    print(f"Chain depth: {buddy.history.depth() if buddy.history else 0}")

    print("\nResidents:")
    for a in s.residents:
        print(f"  - {a.info()}")
