/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref wrap_angles function.
 */

#ifndef OPENKALMAN_WRAP_ANGLES_HPP
#define OPENKALMAN_WRAP_ANGLES_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<wrappable Arg>
#else
  template<typename Arg, std::enable_if_t<wrappable<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg)
  {
    if constexpr (dynamic_dimension<Arg, 0> and not euclidean_vector_space_descriptor<vector_space_descriptor_of_t<Arg, 0>>)
      if (not get_vector_space_descriptor_is_euclidean(get_vector_space_descriptor<0>(arg)))
        throw std::domain_error {"In wrap_angles, specified vector space descriptor does not match that of the object's index 0"};
    using Interface = interface::library_interface<std::decay_t<Arg>>;

    if constexpr (euclidean_vector_space_descriptor<vector_space_descriptor_of_t<Arg, 0>> or identity_matrix<Arg> or zero<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::wrap_angles_defined_for<Arg, Arg&&>)
    {
      return Interface::wrap_angles(std::forward<Arg>(arg), get_vector_space_descriptor<0>(arg));
    }
    else
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_wrappable(arg))
        throw std::domain_error {"Argument of wrap_angles is not wrappable"};

      return from_euclidean(to_euclidean(std::forward<Arg>(arg), get_vector_space_descriptor<0>(arg)), get_vector_space_descriptor<0>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_WRAP_ANGLES_HPP
