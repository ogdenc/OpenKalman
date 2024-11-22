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
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/static_concatenate.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"


namespace OpenKalman::descriptor::internal
{
  /**
   * \brief Reduce a \ref static_vector_space_descriptor into its expanded canonical form.
   * \details By definition, if two descriptors have the same canonical form, they are equivalent.
   * \sa \ref static_canonical_form_t, \ref maybe_equivalent_to, \ref equivalent_to
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct static_canonical_form;


  /**
   * \brief Helper template for \ref static_canonical_form.
   */
  template<typename T>
  using static_canonical_form_t = typename static_canonical_form<T>::type;


  namespace detail
  {
    template<typename T>
    struct consolidate_euclidean_descriptors;


    template<>
    struct consolidate_euclidean_descriptors<StaticDescriptor<>>
    {
      using type = StaticDescriptor<>;
    };


    template<typename C>
    struct consolidate_euclidean_descriptors<StaticDescriptor<C>>
    {
      using type = C;
    };


    template<typename C0, typename C1, typename...Cs>
    struct consolidate_euclidean_descriptors<StaticDescriptor<C0, C1, Cs...>>
    {
      using type = std::conditional_t<
        euclidean_vector_space_descriptor<C0> and euclidean_vector_space_descriptor<C1>,
        static_canonical_form_t<
          StaticDescriptor<Dimensions<dimension_size_of_v<C0> + dimension_size_of_v<C1>>, Cs...>>,
        std::conditional_t<
          euclidean_vector_space_descriptor<C0>,
          static_concatenate_t<
            StaticDescriptor<C0, C1>,
            typename consolidate_euclidean_descriptors<StaticDescriptor<Cs...>>::type>,
          static_concatenate_t<
            StaticDescriptor<C0>,
            typename consolidate_euclidean_descriptors<StaticDescriptor<C1, Cs...>>::type>>>;

    };
  } // namespace detail


#ifdef __cpp_concepts
  template<static_vector_space_descriptor C>
#else
  template<typename C, typename>
#endif
  struct static_canonical_form
  {
    using type = std::conditional_t<
      euclidean_vector_space_descriptor<C>,
      std::conditional_t<
        dimension_size_of_v<C> == 0,
        StaticDescriptor<>,
        Dimensions<dimension_size_of_v<C>>>,
      C>;
  };


  template<typename...Cs>
  struct static_canonical_form<StaticDescriptor<StaticDescriptor<Cs...>>>
  {
    using type = static_canonical_form_t<StaticDescriptor<Cs...>>;
  };


  template<typename...Cs>
  struct static_canonical_form<StaticDescriptor<Cs...>>
  {
    using type = typename detail::consolidate_euclidean_descriptors<
      static_concatenate_t<static_canonical_form_t<Cs>...>>::type;
  };


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_DESCRIPTORS_STATIC_CANONICAL_FORM_HPP
