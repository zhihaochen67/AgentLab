def quantity_for(inventory, sku):
    """Return a SKU quantity, treating a missing SKU as zero."""
    return inventory.get(sku)
