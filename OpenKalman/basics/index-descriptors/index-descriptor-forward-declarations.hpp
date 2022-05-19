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
  // -------------------- //
  //   index_descriptor   //
  // -------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_index_descriptor : std::false_type {};

    template<typename T>
    struct is_index_descriptor<T, std::enable_if_t<
      std::is_convertible<decltype(interface::IndexDescriptorSize<T>::value), std::size_t>::value and
      std::is_convertible<decltype(interface::IndexDescriptorSize<T>::get(std::declval<const std::decay_t<T>&>())), std::size_t>::value and
      std::is_convertible<decltype(interface::EuclideanIndexDescriptorSize<T>::value), std::size_t>::value and
      std::is_convertible<decltype(interface::EuclideanIndexDescriptorSize<T>::get(std::declval<const std::decay_t<T>&>())), std::size_t>::value and
      std::is_void<std::void_t<typename interface::IndexDescriptorDifferenceType<T>::type>>::value>> : std::true_type {};
  }
#endif


  /**
   * \brief A descriptor for a tensor index.
   * \details For T to be an index_descriptor, the following must be defined:
   * - <code>\ref interface::IndexDescriptorSize "interface::IndexDescriptorSize<T>::value"</code>;
   * - <code>\ref interface::IndexDescriptorSize "interface::IndexDescriptorSize<T>::get(t)"</code>
   * where <code>t</code> is an instance of <code>T</code>;
   * - <code>\ref interface::EuclideanIndexDescriptorSize "interface::EuclideanIndexDescriptorSize<T>"</code>; and
   * - <code>\ref interface::EuclideanIndexDescriptorSize "interface::EuclideanIndexDescriptorSize<T>::get(t)"</code>
   * where <code>t</code> is an instance of <code>T</code>;
   * - <code>\ref interface::IndexDescriptorComponentCount "interface::IndexDescriptorComponentCount<T>::value"</code>;
   * - <code>\ref interface::IndexDescriptorComponentCount "interface::IndexDescriptorComponentCount<T>::get(t)"</code>
   * where <code>t</code> is an instance of <code>T</code>;
   * - <code>\ref interface::IndexDescriptorDifferenceType "interface::IndexDescriptorDifferenceType<T>::type"</code>.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept index_descriptor = requires(const std::decay_t<T>& t) {
    {interface::IndexDescriptorSize<std::decay_t<T>>::value} -> std::convertible_to<std::size_t>;
    {interface::IndexDescriptorSize<std::decay_t<T>>::get(t)} -> std::convertible_to<std::size_t>;
    {interface::EuclideanIndexDescriptorSize<std::decay_t<T>>::value} -> std::convertible_to<std::size_t>;
    {interface::EuclideanIndexDescriptorSize<std::decay_t<T>>::get(t)} -> std::convertible_to<std::size_t>;
    {interface::IndexDescriptorComponentCount<std::decay_t<T>>::value} -> std::convertible_to<std::size_t>;
    {interface::IndexDescriptorComponentCount<std::decay_t<T>>::get(t)} -> std::convertible_to<std::size_t>;
    typename interface::IndexDescriptorDifferenceType<std::decay_t<T>>::type;
  };
#else
  constexpr bool index_descriptor = detail::is_index_descriptor<std::decay_t<T>>::value;
#endif


  // --------------------- //
  //   dimension_size_of   //
  // --------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor.
   * \details The associated static member <code>value</code> is the size of the index descriptor
   * (or \ref dynamic_size if not known at compile time).
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
  struct dimension_size_of
#else
  template<typename T, typename = void>
  struct dimension_size_of;

  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<
    std::is_convertible<decltype(interface::IndexDescriptorSize<std::decay_t<T>>::value), std::size_t>::value>>
#endif
    : std::integral_constant<std::size_t, interface::IndexDescriptorSize<std::decay_t<T>>::value> {};


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
   * to Euclidean space (or \ref dynamic_size if not known at compile time).
   */
  #ifdef __cpp_concepts
  template<index_descriptor T>
  struct euclidean_dimension_size_of
  #else
  template<typename T, typename = void>
  struct euclidean_dimension_size_of;

  template<typename T>
  struct euclidean_dimension_size_of<T, std::enable_if_t<
    std::is_convertible<decltype(interface::EuclideanIndexDescriptorSize<std::decay_t<T>>::value), std::size_t>::value>>
  #endif
    : std::integral_constant<std::size_t, interface::EuclideanIndexDescriptorSize<std::decay_t<T>>::value> {};


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
   * \details The associated static member <code>value</code> is the number of atomic component parts
   * (or \ref dynamic_size if not known at compile time).
   */
  #ifdef __cpp_concepts
  template<index_descriptor T>
  struct index_descriptor_components_of
  #else
  template<typename T, typename = void>
  struct index_descriptor_components_of;

  template<typename T>
  struct index_descriptor_components_of<T, std::enable_if_t<
    std::is_convertible<decltype(interface::IndexDescriptorComponentCount<std::decay_t<T>>::value), std::size_t>::value>>
  #endif
    : std::integral_constant<std::size_t, interface::IndexDescriptorComponentCount<std::decay_t<T>>::value> {};


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
   * \details The associated static member <code>value</code> is the number of atomic component parts
   * (or \ref dynamic_size if not known at compile time).
   * For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Axis,
   * so if <code>T</code> is Distance, the resulting <code>type</code> will be Axis.
   */
  #ifdef __cpp_concepts
  template<index_descriptor T>
  struct dimension_difference_of
  #else
  template<typename T, typename = void>
  struct dimension_difference_of;

  template<typename T>
  struct dimension_difference_of<T, std::void_t<
    typename interface::IndexDescriptorDifferenceType<std::decay_t<T>>::type>>
  #endif
  {
    /**
     * \brief The difference type.
     */
    using type = typename interface::IndexDescriptorDifferenceType<std::decay_t<T>>::type;
  };


  /**
   * \brief Helper template for \ref dimension_difference_of.
   */
  template<typename T>
  using dimension_difference_of_t = typename dimension_difference_of<std::decay_t<T>>::type;
  // ---------------------------- //
  //   dynamic_index_descriptor   //
  // ---------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_dynamic_coefficients : std::false_type {};

    template<typename T>
    struct is_dynamic_coefficients<T, std::enable_if_t<dimension_size_of<T>::value == dynamic_size>> : std::true_type {};
  }
#endif


  /**
   * \brief A descriptor for a tensor index that is dynamic (defined at run time).
   * \details This includes any built-in std::integral class or any class T for which
   * <code>\ref dimension_size_of "dimension_size_of<T>::value"</code> is \ref dynamic_size.
   * \sa Dimensions, DynamicCoefficients.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept dynamic_index_descriptor = index_descriptor<T> and (dimension_size_of_v<T> == dynamic_size);
#else
  template<typename T>
  constexpr bool dynamic_index_descriptor = index_descriptor<T> and
    detail::is_dynamic_coefficients<std::decay_t<T>>::value;
#endif


  // -------------------------- //
  //   fixed_index_descriptor   //
  // -------------------------- //

  /**
   * \brief A descriptor for a tensor index that is fixed (defined at compile time).
   * \details This includes any \ref index_descriptor that is not a \ref dynamic_index_descriptor
   */
  template<typename C>
#ifdef __cpp_concepts
  concept fixed_index_descriptor =
#else
  constexpr bool fixed_index_descriptor =
#endif
    index_descriptor<C> and (not dynamic_index_descriptor<C>);


  // ------------------------------ //
  //   euclidean_index_descriptor   //
  // ------------------------------ //

  /**
   * \brief A descriptor for a standard index that is untyped and identifies non-modular coordinates in Euclidean space.
   * \details An \ref index_descriptor is Euclidean if each element of the tensor is an unconstrained std::arithmetic
   * type. This would occur, for example, if the underlying scalar value is an unconstrained floating or integral value.
   * In most applications, the index descriptor will be Euclidean.
   * \sa typed_index_descriptor
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_index_descriptor =
#else
  constexpr bool euclidean_index_descriptor =
#endif
    index_descriptor<T> and interface::IndexDescriptorIsUntyped<std::decay_t<T>>::value;


  // -------------------------- //
  //   typed_index_descriptor   //
  // -------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_typed_index_descriptor : std::false_type {};

    template<typename T>
    struct is_typed_index_descriptor<T, std::enable_if_t<
      std::is_invocable_r<double, decltype(T::template to_euclidean_element<double>),
        const std::function<double(std::size_t)>&, std::size_t, std::size_t>::value and
      std::is_invocable_r<double, decltype(T::template from_euclidean_element<double>),
        const std::function<double(std::size_t)>&, std::size_t, std::size_t>::value and
      std::is_invocable_r<double, decltype(T::template wrap_get_element<double>),
        const std::function<double(std::size_t)>&, std::size_t, std::size_t>::value and
      std::is_invocable<decltype(T::template wrap_set_element<double>),
        const std::function<void(double, std::size_t)>&, const std::function<double(std::size_t)>&,
        double, std::size_t, std::size_t>::value>>
    : std::true_type {};
  }
#endif


  /**
   * \brief A \ref fixed_index_descriptor that is typed (i.e., it can be modular or otherwise constrained in some way).
   * \details Every typed_index_descriptor must define the following member functions:
   * \code
   * // Maps an element to coordinates in Euclidean space.
   * template<typename Scalar>
   * static constexpr Scalar to_euclidean_element(const std::function<Scalar(std::size_t)>& g,
   *   std::size_t euclidean_local_index, std::size_t start);
   *
   * // Maps a coordinate in Euclidean space to an element.
   * template<typename Scalar>
   * static constexpr Scalar from_euclidean_element(const std::function<Scalar(std::size_t)>& g,
   *   std::size_t local_index, std::size_t euclidean_start);
   *
   * // Performs modular wrapping of an element.
   * template<typename Scalar>
   * static constexpr Scalar wrap_get_element(const std::function<Scalar(std::size_t)>& g,
   *   std::size_t local_index, std::size_t start)
   *
   * // Sets an element and then perform any necessary modular wrapping.
   * template<typename Scalar>
   * static constexpr void wrap_set_element(const std::function<void(Scalar, std::size_t)>& s,
   *   const std::function<Scalar(std::size_t)>& g, Scalar x, std::size_t local_index, std::size_t start)
   * \endcode
   * \sa Dimensions, Distance, Angle, Inclination, Polar, Spherical, TypedIndex
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_index_descriptor = fixed_index_descriptor<T> and
    requires(const std::function<void(double, std::size_t)>& s, const std::function<double(std::size_t)>& g,
        double x, std::size_t local, std::size_t start) {
      {std::decay_t<T>::template to_euclidean_element<double>(g, local, start)} -> std::convertible_to<double>;
      {std::decay_t<T>::template from_euclidean_element<double>(g, local, start)} -> std::convertible_to<double>;
      {std::decay_t<T>::template wrap_get_element<double>(g, local, start)} -> std::convertible_to<double>;
      std::decay_t<T>::template wrap_set_element<double>(s, g, x, local, start);
    };
#else
  constexpr bool typed_index_descriptor = fixed_index_descriptor<T> and
    detail::is_typed_index_descriptor<std::decay_t<T>>::value;
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
   * \brief Reduce a \ref typed_index_descriptor into its canonical form.
   * \sa reduced_typed_index_descriptor_t
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct canonical_fixed_index_descriptor;


  // Definition is in index-descriptor-traits.hpp


  /**
   * \brief Helper template for \ref reduced_typed_index_descriptor.
   */
  template<typename T>
  using canonical_fixed_index_descriptor_t = typename canonical_fixed_index_descriptor<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
