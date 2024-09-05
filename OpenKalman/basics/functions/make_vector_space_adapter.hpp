/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_vector_space_adapter.
 */

#ifndef OPENKALMAN_MAKE_VECTOR_SPACE_ADAPTER_HPP
#define OPENKALMAN_MAKE_VECTOR_SPACE_ADAPTER_HPP

namespace OpenKalman
{
  /**
   * \brief If necessary, wrap an object in a wrapper that adds vector space descriptors for each index.
   * \details Any vector space descriptors in the argument are overwritten.
   * \tparam Arg An \ref indexible object.
   * \taram Ds A set of \ref vector_space_descriptor objects
   */
#ifdef __cpp_concepts
  template<indexible Arg, vector_space_descriptor...Ds> requires
    internal::not_more_fixed_than<Arg, Ds...> and (not internal::less_fixed_than<Arg, Ds...>) and
    internal::maybe_same_shape_as_vector_space_descriptors<Arg, Ds...>
#else
  template<typename Arg, typename...Ds, std::enable_if_t<indexible<Arg> and (... and fixed_vector_space_descriptor<Ds>) and
    internal::not_more_fixed_than<Arg, Ds...> and (not internal::less_fixed_than<Arg, Ds...>) and
    internal::maybe_same_shape_as_vector_space_descriptors<Arg, Ds...>, int> = 0>
#endif
  inline auto make_vector_space_adapter(Arg&& arg, Ds&&...ds)
  {
    if constexpr (compatible_with_vector_space_descriptors<Arg, Ds...> and not (... or dynamic_vector_space_descriptor<Ds>))
      return std::forward<Arg>(arg);
    else
      return VectorSpaceAdapter {std::forward<Arg>(arg), std::forward<Ds>(ds)...};
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor...Ds, indexible Arg> requires (not has_dynamic_dimensions<Arg>) and
    (sizeof...(Ds) > 0) and internal::maybe_same_shape_as_vector_space_descriptors<Arg, Ds...>
#else
  template<typename...Ds, typename Arg, std::enable_if_t<
    indexible<Arg> and (... and fixed_vector_space_descriptor<Ds>) and
    (not has_dynamic_dimensions<Arg>) and (sizeof...(Ds) > 0) and
    internal::maybe_same_shape_as_vector_space_descriptors<Arg, Ds...>, int> = 0>
#endif
  inline auto make_vector_space_adapter(Arg&& arg)
  {
    return make_vector_space_adapter(std::forward<Arg>(arg), Ds{}...);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_VECTOR_SPACE_ADAPTER_HPP
