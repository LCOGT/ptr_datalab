def variable_star(input: dict):
  """
  Function to perform variable star analysis on a given image.
  Args:
      input (dict): Input dictionary containing star ra/dec and basenames
  """

  # Input

  coords = input.get("coords")
  basenames = input.get("basenames")

  print(coords.get("ra"), coords.get("dec"))
  print(basenames)

  # Fetch CAT data for basenames from archive

  # Pass CAT data to astrosource

  # Perform variable star analysis using astrosource

  # Return results
