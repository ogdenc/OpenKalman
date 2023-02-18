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
 * \brief Forward definitions for index descriptors.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP

#include <type_traits>

namespace OpenKalman
{
  // --------------------------------------------- //
  //   static_index_value, static_index_value_of   //
  // --------------------------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_static_index_value : std::false_type {};

    template<typename T>
    struct is_static_index_value<T, std::enable_if_t<std::is_convertible<
      decltype(std::decay_t<T>::value), const std::size_t>::value and std::is_default_constructible_v<std::decay_t<T>>>>
      : std::bool_constant<std::is_convertible_v<T, const decltype(std::decay_t<T>::value)> and std::decay_t<T>::value >= 0> {};
  }
#endif


  /**
   * \brief T is a static index value.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept static_index_value = std::convertible_to<decltype(std::decay_t<T>::value), const std::size_t> and
    std::convertible_to<T, const decltype(std::decay_t<T>::value)> and
    (std::decay_t<T>::value >= 0) and std::default_initializable<std::decay_t<T>>;
#else
  constexpr bool static_index_value = detail::is_static_index_value<T>::value;
#endif


  /**
   * \brief The numerical value of a \ref static_index_value.
   * \details If T is not a static, compile-time constant, the result is \ref dynamic_size.
   * \todo Replace this in all its instances by direct conversion to the integral type?
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct static_index_value_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<static_index_value T>
  struct static_index_value_of<T>
#else
  template<typename T>
  struct static_index_value_of<T, std::enable_if_t<static_index_value<T>>>
#endif
    : std::integral_constant<std::size_t, std::decay_t<T>::value> {};


  /**
   * \brief Helper template for \ref static_index_value_of.
   */
  template<typename T>
  constexpr auto static_index_value_of_v = static_index_value_of<T>::value;


  // ----------------------- //
  //   dynamic_index_value   //
  // ----------------------- //

  /**
   * \brief T is a dynamic index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept dynamic_index_value = std::integral<std::decay_t<T>>;
#else
  template<typename T>
  constexpr bool dynamic_index_value = std::is_integral_v<std::decay_t<T>>;
#endif


  // --------------- //
  //   index_value   //
  // --------------- //

  /**
   * \brief T is an index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept index_value =
#else
  template<typename T>
  constexpr bool index_value =
#endif
    static_index_value<T> or dynamic_index_value<T>;


  // -------------------------- //
  //   fixed_index_descriptor   //
  // -------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_fixed_index_descriptor : std::false_type {};

    template<typename T>
    struct is_fixed_index_descriptor<T, std::enable_if_t<
      std::is_default_constructible<std::decay_t<T>>::value and
      std::is_convertible<decltype(interface::FixedIndexDescriptorTraits<std::decay_t<T>>::size), std::size_t>::value and
      std::is_convertible<decltype(interface::FixedIndexDescriptorTraits<std::decay_t<T>>::euclidean_size), std::size_t>::value and
      std::is_convertible<decltype(interface::FixedIndexDescriptorTraits<std::decay_t<T>>::component_count), std::size_t>::value and
      std::is_convertible<decltype(interface::FixedIndexDescriptorTraits<std::decay_t<T>>::always_euclidean), bool>::value and
      std::is_void<std::void_t<typename interface::FixedIndexDescriptorTraits<std::decay_t<T>>::difference_type>>::value
      >> : std::true_type {};
  }
#endif


  /**
   * \brief An \ref index_descriptor for which the number of dimensions is fixed at compile time.
   * \details This includes any object for which interface::FixedIndexDescriptorTraits is defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept fixed_index_descriptor = std::default_initializable<std::decay_t<T>> and
    requires(interface::FixedIndexDescriptorTraits<std::decay_t<T>> t) {
      {decltype(t)::size} -> std::convertible_to<std::size_t>;
      {decltype(t)::euclidean_size} -> std::convertible_to<std::size_t>;
      {decltype(t)::component_count} -> std::convertible_to<std::size_t>;
      {decltype(t)::always_euclidean} -> std::convertible_to<bool>;
      typename decltype(t)::difference_type;
    };
#else
  constexpr bool fixed_index_descriptor = detail::is_fixed_index_descriptor<std::decay_t<T>>::value;
#endif


  // ---------------------------- //
  //   dynamic_index_descriptor   //
  // ---------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_dynamic_index_descriptor : std::false_type {};

    template<typename T>
    struct is_dynamic_index_descriptor<T, std::enable_if_t<
      std::is_convertible<decltype(std::declval<interface::DynamicIndexDescriptorTraits<std::decay_t<T>>>().get_size()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::DynamicIndexDescriptorTraits<std::decay_t<T>>>().get_euclidean_size()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::DynamicIndexDescriptorTraits<std::decay_t<T>>>().get_component_count()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::DynamicIndexDescriptorTraits<std::decay_t<T>>>().is_euclidean()), bool>::value
      >> : std::true_type {};
  }
#endif


  /**
   * \brief An \ref index_descriptor for which the number of dimensions is defined at runtime.
   * \details This includes any object for which interface::DynamicIndexDescriptorTraits is defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_index_descriptor =
    requires(const interface::DynamicIndexDescriptorTraits<std::decay_t<T>>& t) {
      {t.get_size()} -> std::convertible_to<std::size_t>;
      {t.get_euclidean_size()} -> std::convertible_to<std::size_t>;
      {t.get_component_count()} -> std::convertible_to<std::size_t>;
      {t.is_euclidean()} -> std::convertible_to<bool>;
    };
#else
  constexpr bool dynamic_index_descriptor = detail::is_dynamic_index_descriptor<std::decay_t<T>>::value;
#endif


  // -------------------- //
  //   index_descriptor   //
  // -------------------- //

  /**
   * \brief A descriptor for a tensor index, defining the number of dimensions and whether each dimension is modular.
   * \details This includes anything that is either a \ref fixed_index_descriptor or a \ref dynamic_index_descriptor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept index_descriptor =
#else
  constexpr bool index_descriptor =
#endif
    fixed_index_descriptor<T> or dynamic_index_descriptor<T>;


  // --------------------- //
  //   dimension_size_of   //
  // --------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor.
   * \details The associated static member <code>value</code> is the size of the index descriptor,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_size_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<fixed_index_descriptor T>
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<fixed_index_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, interface::FixedIndexDescriptorTraits<std::decay_t<T>>::size> {};


  /**
   * \brief Helper template for \ref dimension_size_of.
   */
  template<typename T>
  constexpr auto dimension_size_of_v = dimension_size_of<T>::value;


  // ------------------------------- //
  //   euclidean_dimension_size_of   //
  // ------------------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor if it is transformed into Euclidean space.
   * \details The associated static member <code>value</code> is the size of the index descriptor when transformed
   * to Euclidean space, or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct euclidean_dimension_size_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<fixed_index_descriptor T>
  struct euclidean_dimension_size_of<T>
#else
  template<typename T>
  struct euclidean_dimension_size_of<T, std::enable_if_t<fixed_index_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, interface::FixedIndexDescriptorTraits<std::decay_t<T>>::euclidean_size> {};


  /**
   * \brief Helper template for \ref euclidean_dimension_size_of.
   */
  template<typename T>
  constexpr auto euclidean_dimension_size_of_v = euclidean_dimension_size_of<T>::value;


  // ---------------------------------- //
  //   index_descriptor_components_of   //
  // ---------------------------------- //

  /**
   * \brief The number of atomic component parts of an \ref index_descriptor.
   * \details The associated static member <code>value</code> is the number of atomic component parts,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct index_descriptor_components_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<fixed_index_descriptor T>
  struct index_descriptor_components_of<T>
#else
  template<typename T>
  struct index_descriptor_components_of<T, std::enable_if_t<fixed_index_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, interface::FixedIndexDescriptorTraits<std::decay_t<T>>::component_count> {};


  /**
   * \brief Helper template for \ref index_descriptor_components_of.
   */
  template<typename T>
  constexpr auto index_descriptor_components_of_v = index_descriptor_components_of<T>::value;


  // --------------------------- //
  //   dimension_difference_of   //
  // --------------------------- //

  /**
   * \brief The type of the \ref index_descriptor when tensors having respective index_descriptors T are subtracted.
   * \details The associated alias <code>type</code> is the difference type.
   * For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Axis,
   * so if <code>T</code> is Distance, the resulting <code>type</code> will be Dimensions<1>.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_difference_of { using type = std::decay_t<T>; };


#ifdef __cpp_concepts
  template<fixed_index_descriptor T>
  struct dimension_difference_of<T>
#else
  template<typename T>
  struct dimension_difference_of<T, std::enable_if_t<fixed_index_descriptor<T>>>
#endif
  { using type = typename interface::FixedIndexDescriptorTraits<std::decay_t<T>>::difference_type; };


  /**
   * \brief Helper template for \ref dimension_difference_of.
   */
  template<typename T>
  using dimension_difference_of_t = typename dimension_difference_of<std::decay_t<T>>::type;


  // ------------------------------ //
  //   euclidean_index_descriptor   //
  // ------------------------------ //

  /**
   * \brief A descriptor for a normal tensor index, which identifies non-modular coordinates in Euclidean space.
   * \details An \ref index_descriptor is Euclidean if each element of the tensor is an unconstrained std::arithmetic
   * type. This would occur, for example, if the underlying scalar value is an unconstrained floating or integral value.
   * In most applications, the index descriptor will be Euclidean.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_index_descriptor = index_descriptor<T> and
    (interface::FixedIndexDescriptorTraits<std::decay_t<T>>::always_euclidean or
    interface::DynamicIndexDescriptorTraits<std::decay_t<T>>::is_euclidean());
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_euclidean_index_descriptor : std::false_type {};

    template<typename T>
    struct is_euclidean_index_descriptor<T, std::enable_if_t<fixed_index_descriptor<T>>>
      : std::bool_constant<interface::FixedIndexDescriptorTraits<T>::always_euclidean> {};

    template<typename T>
    struct is_euclidean_index_descriptor<T, std::enable_if_t<index_descriptor<T> and not fixed_index_descriptor<T> and
      interface::DynamicIndexDescriptorTraits<std::decay_t<T>>::is_euclidean()>> : std::true_type {};
  }

  template<typename T>
  constexpr bool euclidean_index_descriptor = detail::is_euclidean_index_descriptor<std::decay_t<T>>::value;
#endif


  // -------------------------------------- //
  //   concatenate_fixed_index_descriptor   //
  // -------------------------------------- //

  /**
   * \brief Concatenate any number of TypedIndex<...> types.
   * \details Example: \code concatenate_fixed_index_descriptor_t<TypedIndex<angle::Radians>, TypedIndex<Axis, Distance>> ==
   * TypedIndex<angle::Radians, Axis, Distance> \endcode.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor ... Cs>
#else
  template<typename ... Cs>
#endif
  struct concatenate_fixed_index_descriptor;


  // Definition is in index-descriptor-traits.hpp


  /**
   * \brief Helper template for \ref concatenate_fixed_index_descriptor.
   */
  template<typename...Cs>
  using concatenate_fixed_index_descriptor_t = typename concatenate_fixed_index_descriptor<Cs...>::type;


  // ------------------------------------ //
  //   canonical_fixed_index_descriptor   //
  // ------------------------------------ //

  /**
   * \brief Reduce a \ref fixed_index_descriptor into its canonical form.
   * \sa canonical_fixed_index_descriptor_t
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct canonical_fixed_index_descriptor;


  // Definition is in index-descriptor-traits.hpp


  /**
   * \brief Helper template for \ref canonical_fixed_index_descriptor.
   */
  template<typename T>
  using canonical_fixed_index_descriptor_t = typename canonical_fixed_index_descriptor<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
