/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref static_canonical_form.
 */

#ifndef OPENKALMAN_DESCRIPTORS_STATIC_CANONICAL_FORM_HPP
#define OPENKALMAN_DESCRIPTORS_STATIC_CANONICAL_FORM_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/static_concatenate.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/canonical_equivalent.hpp"


namespace OpenKalman::descriptor::internal
{
  namespace detail
  {
    template<typename A>
    struct consolidate_descriptors { using type = A; };


    template<typename A>
    struct consolidate_descriptors<StaticDescriptor<A>> { using type = A; };


    template<typename A>
    using consolidate_descriptors_t = typename consolidate_descriptors<A>::type;


    template<typename A0, typename A1, typename...As>
    struct consolidate_descriptors<StaticDescriptor<A0, A1, As...>>
    {
      using type = std::conditional_t<
        euclidean_vector_space_descriptor<A0> and euclidean_vector_space_descriptor<A1>,
        consolidate_descriptors_t<StaticDescriptor<Dimensions<dimension_size_of_v<A0> + dimension_size_of_v<A1>>, As...>>,
        std::conditional_t<
          euclidean_vector_space_descriptor<A0>,
          static_concatenate_t<StaticDescriptor<A0, A1>, consolidate_descriptors_t<StaticDescriptor<As...>>>,
          static_concatenate_t<A0, consolidate_descriptors_t<StaticDescriptor<A1, As...>>>>>;

    };
  } // namespace detail


  /**
   * \internal
   * \brief Reduce a \ref static_vector_space_descriptor into its expanded canonical form.
   * \details By definition, if two descriptors have the same canonical form, they are equivalent.
   * \sa \ref static_canonical_form_t, \ref maybe_equivalent_to, \ref equivalent_to
   */
  template<typename T>
  struct static_canonical_form
  {
    using type = std::decay_t<decltype(descriptor::internal::canonical_equivalent(std::declval<T>()))>;
  };


  /**
   * \internal
   * \brief Helper template for \ref static_canonical_form.
   */
  template<typename T>
  using static_canonical_form_t = typename static_canonical_form<T>::type;


  template<typename...Cs>
  struct static_canonical_form<StaticDescriptor<Cs...>>
  {
    using type = detail::consolidate_descriptors_t<static_concatenate_t<static_canonical_form_t<Cs>...>>;
  };


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_DESCRIPTORS_STATIC_CANONICAL_FORM_HPP
