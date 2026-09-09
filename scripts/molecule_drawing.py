"""Aligned, publication-scale rendering of a molecular transformation."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


EDIT_COLOR = (0.90, 0.49, 0.08)
EDIT_FILL = (1.00, 0.88, 0.70)


def aligned_molecules(left_smiles: str, right_smiles: str):
    left = Chem.MolFromSmiles(left_smiles)
    right = Chem.MolFromSmiles(right_smiles)
    if left is None or right is None:
        raise ValueError("Invalid SMILES in molecular pair")
    result = rdFMCS.FindMCS(
        [left, right], timeout=10, ringMatchesRingOnly=True,
        completeRingsOnly=True, atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
    )
    common = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    rdDepictor.Compute2DCoords(left)
    if common is not None:
        try:
            rdDepictor.GenerateDepictionMatching2DStructure(
                right, left, refPatt=common, acceptFailure=True,
            )
        except Exception:
            rdDepictor.Compute2DCoords(right)
    else:
        rdDepictor.Compute2DCoords(right)

    left_match = set(left.GetSubstructMatch(common)) if common is not None else set()
    right_match = set(right.GetSubstructMatch(common)) if common is not None else set()
    left_atoms = [i for i in range(left.GetNumAtoms()) if i not in left_match]
    right_atoms = [i for i in range(right.GetNumAtoms()) if i not in right_match]

    def edited_bonds(mol, retained):
        return [bond.GetIdx() for bond in mol.GetBonds()
                if bond.GetBeginAtomIdx() not in retained or bond.GetEndAtomIdx() not in retained]

    return left, right, left_atoms, right_atoms, edited_bonds(left, left_match), edited_bonds(right, right_match)


def _crop(image: Image.Image, padding: int = 12) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    mask = np.any(array < 248, axis=2)
    if not mask.any():
        return image
    yy, xx = np.where(mask)
    return image.crop((max(0, xx.min()-padding), max(0, yy.min()-padding),
                       min(array.shape[1], xx.max()+padding+1),
                       min(array.shape[0], yy.max()+padding+1)))


def render_aligned_pair(left_smiles: str, right_smiles: str,
                        size: tuple[int, int] = (720, 430)) -> list[Image.Image]:
    left, right, left_atoms, right_atoms, left_bonds, right_bonds = aligned_molecules(
        left_smiles, right_smiles
    )
    images = []
    for mol, atoms, bonds in ((left, left_atoms, left_bonds),
                              (right, right_atoms, right_bonds)):
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
        opts.fillHighlights = True
        opts.continuousHighlight = False
        opts.atomHighlightsAreCircles = True
        opts.highlightRadius = 0.24
        opts.bondLineWidth = 2.0
        opts.highlightBondWidthMultiplier = 10
        opts.minFontSize = 13
        opts.maxFontSize = 24
        atom_colors = {idx: EDIT_FILL for idx in atoms}
        bond_colors = {idx: EDIT_COLOR for idx in bonds}
        drawer.DrawMolecule(mol, highlightAtoms=atoms, highlightBonds=bonds,
                            highlightAtomColors=atom_colors,
                            highlightBondColors=bond_colors)
        drawer.FinishDrawing()
        images.append(_crop(Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")))
    return images


def render_aligned_pair_grid(left_smiles: str, right_smiles: str,
                             panel_size: tuple[int, int] = (560, 360)) -> Image.Image:
    """Render both molecules on a single same-scale canvas."""
    left, right, left_atoms, right_atoms, left_bonds, right_bonds = aligned_molecules(
        left_smiles, right_smiles
    )
    width, height = panel_size
    drawer = rdMolDraw2D.MolDraw2DCairo(2*width, height, width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    opts.fillHighlights = True
    opts.continuousHighlight = False
    opts.atomHighlightsAreCircles = True
    opts.highlightRadius = 0.22
    opts.bondLineWidth = 2.0
    opts.highlightBondWidthMultiplier = 10
    opts.minFontSize = 13
    opts.maxFontSize = 24
    opts.drawMolsSameScale = True
    drawer.DrawMolecules(
        [left, right],
        highlightAtoms=[left_atoms, right_atoms],
        highlightBonds=[left_bonds, right_bonds],
        highlightAtomColors=[{idx: EDIT_FILL for idx in left_atoms},
                             {idx: EDIT_FILL for idx in right_atoms}],
        highlightBondColors=[{idx: EDIT_COLOR for idx in left_bonds},
                             {idx: EDIT_COLOR for idx in right_bonds}],
    )
    drawer.FinishDrawing()
    return _crop(Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB"), padding=6)


def render_aligned_triplet_grid(
    query_smiles: str, neighbor_smiles: list[str],
    panel_size: tuple[int, int] = (1080, 540),
) -> Image.Image:
    """Render one query and two aligned neighbors in monochrome at one scale."""
    if len(neighbor_smiles) != 2:
        raise ValueError("triplet rendering requires exactly two neighbors")
    query = Chem.MolFromSmiles(query_smiles)
    neighbors = [Chem.MolFromSmiles(value) for value in neighbor_smiles]
    if query is None or any(mol is None for mol in neighbors):
        raise ValueError("Invalid SMILES in molecular triplet")
    rdDepictor.Compute2DCoords(query)
    for neighbor in neighbors:
        result = rdFMCS.FindMCS(
            [query, neighbor], timeout=10, ringMatchesRingOnly=True,
            completeRingsOnly=True, atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        common = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
        if common is not None:
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(
                    neighbor, query, refPatt=common, acceptFailure=True,
                )
            except Exception:
                rdDepictor.Compute2DCoords(neighbor)
        else:
            rdDepictor.Compute2DCoords(neighbor)

    width, height = panel_size
    drawer = rdMolDraw2D.MolDraw2DCairo(3 * width, height, width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    opts.useBWAtomPalette()
    opts.bondLineWidth = 1.7
    opts.minFontSize = 22
    opts.maxFontSize = 34
    opts.padding = 0.10
    opts.drawMolsSameScale = True
    drawer.DrawMolecules([query, *neighbors])
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
