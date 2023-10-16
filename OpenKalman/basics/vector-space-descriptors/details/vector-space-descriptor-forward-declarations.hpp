/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward definitions for \ref vector_space_descriptor types.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP

#include <type_traits>

namespace OpenKalman
{
  // -------------------------- //
  //   fixed_vector_space_descriptor   //
  // -------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_fixed_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_fixed_vector_space_descriptor<T, std::enable_if_t<
      std::is_default_constructible<std::decay_t<T>>::value and
      std::is_convertible<decltype(interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::size), std::size_t>::value and
      std::is_convertible<decltype(interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::euclidean_size), std::size_t>::value and
      std::is_convertible<decltype(interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::component_count), std::size_t>::value and
      std::is_convertible<decltype(interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::always_euclidean), bool>::value and
      std::is_void<std::void_t<typename interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::difference_type>>::value
      >> : std::true_type {};
  }
#endif


  /**
   * \brief A set of \ref vector_space_descriptor for which the number of dimensions is fixed at compile time.
   * \details This includes any object for which interface::FixedVectorSpaceDescriptorTraits is defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept fixed_vector_space_descriptor = std::default_initializable<std::decay_t<T>> and
    requires(interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>> t) {
      {decltype(t)::size} -> std::convertible_to<std::size_t>;
      {decltype(t)::euclidean_size} -> std::convertible_to<std::size_t>;
      {decltype(t)::component_count} -> std::convertible_to<std::size_t>;
      {decltype(t)::always_euclidean} -> std::convertible_to<bool>;
      typename decltype(t)::difference_type;
    };
#else
  constexpr bool fixed_vector_space_descriptor = detail::is_fixed_vector_space_descriptor<std::decay_t<T>>::value;
#endif


  // ---------------------------- //
  //   dynamic_vector_space_descriptor   //
  // ---------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_dynamic_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_dynamic_vector_space_descriptor<T, std::enable_if_t<
      std::is_convertible<decltype(std::declval<interface::DynamicVectorSpaceDescriptorTraits<std::decay_t<T>>>().get_size()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::DynamicVectorSpaceDescriptorTraits<std::decay_t<T>>>().get_euclidean_size()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::DynamicVectorSpaceDescriptorTraits<std::decay_t<T>>>().get_component_count()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::DynamicVectorSpaceDescriptorTraits<std::decay_t<T>>>().is_euclidean()), bool>::value
      >> : std::true_type {};
  }
#endif


  /**
   * \brief A set of \ref vector_space_descriptor for which the number of dimensions is defined at runtime.
   * \details This includes any object for which interface::DynamicVectorSpaceDescriptorTraits is defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_vector_space_descriptor =
    requires(const interface::DynamicVectorSpaceDescriptorTraits<std::decay_t<T>>& t) {
      {t.get_size()} -> std::convertible_to<std::size_t>;
      {t.get_euclidean_size()} -> std::convertible_to<std::size_t>;
      {t.get_component_count()} -> std::convertible_to<std::size_t>;
      {t.is_euclidean()} -> std::convertible_to<bool>;
    };
#else
  constexpr bool dynamic_vector_space_descriptor = detail::is_dynamic_vector_space_descriptor<std::decay_t<T>>::value;
#endif


  // -------------------- //
  //   vector_space_descriptor   //
  // -------------------- //

  /**
   * \brief An object describing the type of (vector) space associated with a tensor index.
   * \details Such an object is a trait defining the number of dimensions and whether each dimension is modular.
   * This includes anything that is either a \ref fixed_vector_space_descriptor or a \ref dynamic_vector_space_descriptor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept vector_space_descriptor =
#else
  constexpr bool vector_space_descriptor =
#endif
    fixed_vector_space_descriptor<T> or dynamic_vector_space_descriptor<T>;


  // --------------------- //
  //   dimension_size_of   //
  // --------------------- //

  /**
   * \brief The dimension size of a set of \ref vector_space_descriptor.
   * \details The associated static member <code>value</code> is the size of the \ref vector_space_descriptor,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_size_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::size> {};


  /**
   * \brief Helper template for \ref dimension_size_of.
   */
  template<typename T>
  constexpr auto dimension_size_of_v = dimension_size_of<T>::value;


  // ------------------------------- //
  //   euclidean_dimension_size_of   //
  // ------------------------------- //

  /**
   * \brief The dimension size of a set of \ref vector_space_descriptor if it is transformed into Euclidean space.
   * \details The associated static member <code>value</code> is the size of the \ref vector_space_descriptor when transformed
   * to Euclidean space, or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct euclidean_dimension_size_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct euclidean_dimension_size_of<T>
#else
  template<typename T>
  struct euclidean_dimension_size_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::euclidean_size> {};


  /**
   * \brief Helper template for \ref euclidean_dimension_size_of.
   */
  template<typename T>
  constexpr auto euclidean_dimension_size_of_v = euclidean_dimension_size_of<T>::value;


  // ---------------------------------- //
  //   vector_space_descriptor_components_of   //
  // ---------------------------------- //

  /**
   * \brief The number of atomic component parts of a set of \ref vector_space_descriptor.
   * \details The associated static member <code>value</code> is the number of atomic component parts,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct vector_space_descriptor_components_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct vector_space_descriptor_components_of<T>
#else
  template<typename T>
  struct vector_space_descriptor_components_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::component_count> {};


  /**
   * \brief Helper template for \ref vector_space_descriptor_components_of.
   */
  template<typename T>
  constexpr auto vector_space_descriptor_components_of_v = vector_space_descriptor_components_of<T>::value;


  // --------------------------- //
  //   dimension_difference_of   //
  // --------------------------- //

  /**
   * \brief The type of the \ref vector_space_descriptor object when tensors having respective vector_space_descriptor T are subtracted.
   * \details The associated alias <code>type</code> is the difference type.
   * For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Axis,
   * so if <code>T</code> is Distance, the resulting <code>type</code> will be Dimensions<1>.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_difference_of { using type = std::decay_t<T>; };


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct dimension_difference_of<T>
#else
  template<typename T>
  struct dimension_difference_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
  { using type = typename interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::difference_type; };


  /**
   * \brief Helper template for \ref dimension_difference_of.
   */
  template<typename T>
  using dimension_difference_of_t = typename dimension_difference_of<std::decay_t<T>>::type;


  // ------------------------------ //
  //   euclidean_vector_space_descriptor   //
  // ------------------------------ //

  /**
   * \brief A \ref vector_space_descriptor for a normal tensor index, which identifies non-modular coordinates in Euclidean space.
   * \details A set of \ref vector_space_descriptor is Euclidean if each element of the tensor is an unconstrained std::arithmetic
   * type. This would occur, for example, if the underlying scalar value is an unconstrained floating or integral value.
   * In most applications, the \ref vector_space_descriptor will be Euclidean.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_vector_space_descriptor = vector_space_descriptor<T> and
    (interface::FixedVectorSpaceDescriptorTraits<std::decay_t<T>>::always_euclidean or
    interface::DynamicVectorSpaceDescriptorTraits<std::decay_t<T>>::is_euclidean());
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_euclidean_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_euclidean_vector_space_descriptor<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
      : std::bool_constant<interface::FixedVectorSpaceDescriptorTraits<T>::always_euclidean> {};

    template<typename T>
    struct is_euclidean_vector_space_descriptor<T, std::enable_if_t<vector_space_descriptor<T> and not fixed_vector_space_descriptor<T> and
      interface::DynamicVectorSpaceDescriptorTraits<std::decay_t<T>>::is_euclidean()>> : std::true_type {};
  }

  template<typename T>
  constexpr bool euclidean_vector_space_descriptor = detail::is_euclidean_vector_space_descriptor<std::decay_t<T>>::value;
#endif


  // -------------------------------------- //
  //   concatenate_fixed_vector_space_descriptor   //
  // -------------------------------------- //

  /**
   * \brief Concatenate any number of TypedIndex<...> types.
   * \details Example: \code concatenate_fixed_vector_space_descriptor_t<TypedIndex<angle::Radians>, TypedIndex<Axis, Distance>> ==
   * TypedIndex<angle::Radians, Axis, Distance> \endcode.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor ... Cs>
#else
  template<typename ... Cs>
#endif
  struct concatenate_fixed_vector_space_descriptor;


  // Definition is in vector-type-traits.hpp


  /**
   * \brief Helper template for \ref concatenate_fixed_vector_space_descriptor.
   */
  template<typename...Cs>
  using concatenate_fixed_vector_space_descriptor_t = typename concatenate_fixed_vector_space_descriptor<Cs...>::type;


  // ------------------------------------ //
  //   canonical_fixed_vector_space_descriptor   //
  // ------------------------------------ //

  /**
   * \brief Reduce a \ref fixed_vector_space_descriptor into its canonical form.
   * \sa canonical_fixed_vector_space_descriptor_t
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct canonical_fixed_vector_space_descriptor;


  // Definition is in vector-type-traits.hpp


  /**
   * \brief Helper template for \ref canonical_fixed_vector_space_descriptor.
   */
  template<typename T>
  using canonical_fixed_vector_space_descriptor_t = typename canonical_fixed_vector_space_descriptor<T>::type;



  namespace internal
  {
    /**
     * \internal
     * \brief Update only the real part of a (potentially) complex number, leaving the imaginary part unchanged.
     * \param arg A potentially complex number to update.
     * \param re A real value.
     */
  #ifdef __cpp_concepts
    constexpr scalar_type decltype(auto)
    update_real_part(scalar_type auto&& arg, scalar_type auto&& re) requires (not complex_number<decltype(re)>)
  #else
    template<typename T, typename Re, std::enable_if_t<scalar_type<T> and scalar_type<Re> and not complex_number<Re>, int> = 0>
    constexpr decltype(auto) update_real_part(T&& arg, Re&& re)
  #endif
    {
      using Arg = std::decay_t<decltype(arg)>;
      if constexpr (complex_number<Arg>)
      {
        auto im = constexpr_imag(std::forward<decltype(arg)>(arg));
        using R = std::decay_t<decltype(im)>;
        return make_complex_number<Arg>(static_cast<R>(std::forward<decltype(re)>(re)), std::move(im));
      }
      else return std::forward<decltype(re)>(re);
    }

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
