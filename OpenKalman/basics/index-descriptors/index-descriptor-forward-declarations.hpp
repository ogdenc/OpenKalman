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
#include <functional>

namespace OpenKalman
{
  // -------------------- //
  //   index_descriptor   //
  // -------------------- //

#ifndef __cpp_concepts
  /*template<std::size_t> struct Dimensions;
  //struct Axis;
  struct Distance;
  template<template<typename> typename Limits> struct Angle;
  template<template<typename> typename Limits> struct Inclination;
  template<typename, typename, typename> struct Polar;
  template<typename, typename, typename, typename> struct Spherical;
  template<typename...> struct Coefficients;
  template<typename T, std::size_t> struct DynamicCoefficients;*/

  namespace detail
  {
    template<typename T, typename = void>
    struct is_index_descriptor : std::false_type {};

    template<typename T>
    struct is_index_descriptor<T, std::enable_if_t<
      std::is_void<std::void_t<dimension_size_of<T>>>::value //and
      //std::is_void<std::void_t<euclidean_dimension_size_of<T>>>::value and
      //std::is_convertible<decltype(dimension_size_of<T>::value), std::size_t>::value and
      //std::is_convertible<decltype(dimension_size_of<T>::get(std::declval<const std::decay_t<T>&>())), std::size_t>::value and
      //std::is_convertible<decltype(euclidean_dimension_size_of<T>::value), std::size_t>::value and
      //std::is_convertible<decltype(euclidean_dimension_size_of<T>::get(std::declval<const std::decay_t<T>&>())), std::size_t>::value and
      //std::is_void<std::void_t<dimension_difference_of<T>>>::value
      >>
      : std::true_type {};
  }
#endif


  /**
   * \brief A descriptor for a tensor index.
   * \details For T to be an index_descriptor, the following must be defined:
   * - <code>\ref dimension_size_of "dimension_size_of<T>::value"</code>;
   * - <code>\ref dimension_size_of "dimension_size_of<T>::get(t)"</code>
   * where <code>t</code> is an instance of <code>T</code>;
   * - <code>\ref euclidean_dimension_size_of "euclidean_dimension_size_of<T>"</code>; and
   * - <code>\ref euclidean_dimension_size_of "euclidean_dimension_size_of<T>::get(t)"</code>
   * where <code>t</code> is an instance of <code>T</code>;
   * - <code>\ref index_descriptor_components_of "index_descriptor_components_of<T>::value"</code>;
   * - <code>\ref index_descriptor_components_of "index_descriptor_components_of<T>::get(t)"</code>
   * where <code>t</code> is an instance of <code>T</code>;
   * - <code>\ref dimension_difference_of "dimension_difference_of<T>::type"</code>.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept index_descriptor = std::integral<std::decay_t<T>> or requires(const std::decay_t<T>& t) {
    {dimension_size_of<T>::value} -> std::convertible_to<std::size_t>;
    {dimension_size_of<T>::get(t)} -> std::convertible_to<std::size_t>;
    {euclidean_dimension_size_of<T>::value} -> std::convertible_to<std::size_t>;
    {euclidean_dimension_size_of<T>::get(t)} -> std::convertible_to<std::size_t>;
    {index_descriptor_components_of<T>::value} -> std::convertible_to<std::size_t>;
    {index_descriptor_components_of<T>::get(t)} -> std::convertible_to<std::size_t>;
    typename dimension_difference_of<T>::type;
  };
#else
  constexpr bool index_descriptor = std::is_integral_v<std::decay_t<T>> or detail::is_index_descriptor<T>::value;
#endif


  // ---------------------------- //
  //   dynamic_index_descriptor   //
  // ---------------------------- //

#ifndef __cpp_concepts
  template<std::size_t> struct Dimensions;
  template<typename Scalar> struct DynamicCoefficients;

  namespace detail
  {

    template<typename T, typename = void>
    struct is_dynamic_coefficients : std::false_type {};

    //template<typename T>
    //struct is_dynamic_coefficients<T, std::enable_if_t<dimension_size_of<T>::value == dynamic_size>> : std::true_type {};

    template<typename T>
    struct is_dynamic_coefficients<T, std::enable_if_t<std::is_integral_v<T>>> : std::true_type {};

    template<>
    struct is_dynamic_coefficients<Dimensions<dynamic_size>> : std::true_type {};

    template<typename Scalar>
    struct is_dynamic_coefficients<DynamicCoefficients<Scalar>> : std::true_type {};
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
  concept fixed_index_descriptor = index_descriptor<C> and (not dynamic_index_descriptor<C>);
#else
  constexpr bool fixed_index_descriptor = index_descriptor<C> and (not dynamic_index_descriptor<C>);
#endif


  // ------------------------------------------------------- //
  //   Dimensions, Axis, Coefficients, DynamicCoefficients   //
  // ------------------------------------------------------- //

  // Documentation in Dimensions.hpp
  template<std::size_t size>
  struct Dimensions;


  // Documentation in Axis.hpp
  //struct Axis;


  // Documentation in Coefficients.hpp
#ifdef __cpp_concepts
  template<fixed_index_descriptor...Cs>
#else
  template<typename...Cs>
#endif
  struct Coefficients;


  // Documentation in DynamicCoefficients.hpp
  template<typename Scalar>
  struct DynamicCoefficients;


  // ---------------------------- //
  //   untyped_index_descriptor   //
  // ---------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct untyped_index_descriptor_defined : std::false_type {};

    template<typename T>
    struct untyped_index_descriptor_defined<T, std::enable_if_t<std::is_integral_v<T>>> : std::true_type {};

    template<std::size_t N>
    struct untyped_index_descriptor_defined<Dimensions<N>> : std::true_type {};

    template<typename...Cs>
    struct untyped_index_descriptor_defined<Coefficients<Cs...>, std::enable_if_t<
      (untyped_index_descriptor_defined<Cs>::value and ...)>> : std::true_type {};
  }
#endif


  /**
   * \brief A descriptor for an index that is untyped.
   * \details An \ref index_descriptor is untyped if each element of the tensor is an unconstrained std::arithmetic
   * type. This would occur, for example, if the underlying scalar value is an unconstrained floating or integral value.
   * In most applications, the index descriptor will be untyped.
   * \sa typed_index_descriptor
   */
  template<typename T>
#ifdef __cpp_concepts
  concept untyped_index_descriptor =
    index_descriptor<T> and is_untyped_index_descriptor<std::decay_t<T>>::value;
#else
  constexpr bool untyped_index_descriptor =
    index_descriptor<T> and detail::untyped_index_descriptor_defined<std::decay_t<T>>::value;
#endif


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
   * \brief An \ref index_descriptor that is typed (i.e., modular or otherwise constrained in some way).
   * \details Every typed_index_descriptor must also be an \ref index_descriptor_group.
   * <b>Examples:</b>:
   * - angle::Radians
   * - Polar<Distance, angle::Radians>
   * - Coefficients<Axis, inclination::Radians>
   * - Coefficients<Spherical<angle::Degrees, inclination::degrees, Distance>, Axis, Axis>
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_index_descriptor = fixed_index_descriptor<T> and (not untyped_index_descriptor<T>) and
    requires(const std::function<void(double, std::size_t)>& s, const std::function<double(std::size_t)>& g,
        double x, std::size_t local, std::size_t start) {
      {std::decay_t<T>::template to_euclidean_element<double>(g, local, start)} -> std::convertible_to<double>;
      {std::decay_t<T>::template from_euclidean_element<double>(g, local, start)} -> std::convertible_to<double>;
      {std::decay_t<T>::template wrap_get_element<double>(g, local, start)} -> std::convertible_to<double>;
      std::decay_t<T>::template wrap_set_element<double>(s, g, x, local, start);
    };
#else
  constexpr bool typed_index_descriptor = fixed_index_descriptor<T> and (not untyped_index_descriptor<T>) and
    detail::is_typed_index_descriptor<std::decay_t<T>>::value;
#endif


  // ------------------------------ //
  //   composite_index_descriptor   //
  // ------------------------------ //

  namespace detail
  {
    template<typename T>
    struct is_composite_index_descriptor : std::false_type {};

    template<typename...C>
    struct is_composite_index_descriptor<Coefficients<C...>> : std::true_type {};

    template<typename Scalar>
    struct is_composite_index_descriptor<DynamicCoefficients<Scalar>> : std::true_type {};
  }


  /**
   * \brief T is a composite index descriptor.
   * \details A composite index descriptor is a container for other index descriptors, and can either be
   * Coefficients or DynamicCoefficients.
   * \sa Coefficients, DynamicCoefficients.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept composite_index_descriptor =
#else
  constexpr bool composite_index_descriptor =
#endif
    index_descriptor<T> and detail::is_composite_index_descriptor<std::decay_t<T>>::value;


  // --------------------------------- //
  //   atomic_fixed_index_descriptor   //
  // --------------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_untyped_atomic : std::false_type {};

    //template<>
    //struct is_untyped_atomic<Axis> : std::true_type {};

    template<>
    struct is_untyped_atomic<Dimensions<1>> : std::true_type {};

    //template<std::size_t N>
    //struct is_untyped_atomic<Dimensions<N>> : std::bool_constant<N != dynamic_size> {};
  }


  /**
   * \brief T is an atomic (non-separable or non-composite) group of fixed index descriptors.
   * \details These descriptors are suitable for incorporation in
   * \ref composite_index_descriptor "composite index descriptors".
   */
  template<typename T>
#ifdef __cpp_concepts
  concept atomic_fixed_index_descriptor =
#else
  constexpr bool atomic_fixed_index_descriptor =
#endif
    (typed_index_descriptor<T> and (not composite_index_descriptor<T>)) or detail::is_untyped_atomic<T>::value;


  // ------------------------------------- //
  //   replicated_fixed_index_descriptor   //
  // ------------------------------------- //

  namespace detail
  {
    template<typename C, std::size_t...I>
    auto replicate_inds(std::index_sequence<I...>)
    {
      return Coefficients<std::conditional_t<(I==I), C, C>...> {};
    };
  }


  /**
   * \brief Alias for <code>Coefficients<C...></code>, where <code>C</code> is repeated <var>N</var> times.
   * \tparam C The coefficient to be repeated.
   * \tparam N The number of times to repeat coefficient C.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor C, std::size_t N> requires (N != dynamic_size)
#else
  template<typename C, std::size_t N, std::enable_if_t<fixed_index_descriptor<C> and N != dynamic_size, int> = 0>
#endif
  using replicated_fixed_index_descriptor =
    std::conditional_t<N == 1, C, decltype(detail::replicate_inds<std::decay_t<C>>(std::make_index_sequence<N> {}))>;


  // --------------------------------- //
  //   Concatenation of coefficients   //
  // --------------------------------- //

  namespace detail
  {
    template<typename...>
    struct ConcatenateImpl;

    template<>
    struct ConcatenateImpl<>
    {
      using type = Coefficients<>;
    };

    template<typename C, typename...Cs>
    struct ConcatenateImpl<C, Cs...>
    {
      using type = typename ConcatenateImpl<Cs...>::type::template Prepend<C>;
    };

    template<typename...C, typename...Cs>
    struct ConcatenateImpl<Coefficients<C...>, Cs...>
    {
      using type = typename ConcatenateImpl<Cs...>::type::template Prepend<C...>;
    };
  }


  /**
   * \brief Concatenate any number of Coefficients<...> types.
   * \details Example: \code Concatenate<Coefficients<angle::Radians>, Coefficients<Axis, Distance>> ==
   * Coefficients<angle::Radians, Axis, Distance> \endcode.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
#else
  template<typename ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
#endif


  // ---------------------------------- //
  //   reduced_fixed_index_descriptor   //
  // ---------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct reduced_fixed_index_descriptor_impl
    {
      using type = T;
    };


#ifdef __cpp_concepts
    template<std::size_t N> requires (N != dynamic_size)
    struct reduced_fixed_index_descriptor_impl<Dimensions<N>>
#else
    template<std::size_t N>
    struct reduced_fixed_index_descriptor_impl<Dimensions<N>, std::enable_if_t<N != dynamic_size>>
#endif
    {
      using type = std::conditional_t<N == 1, Coefficients<replicated_fixed_index_descriptor<Dimensions<1>, N>>,
        replicated_fixed_index_descriptor<Dimensions<1>, N>>;
    };


#ifdef __cpp_concepts
    template<atomic_fixed_index_descriptor C>
    struct reduced_fixed_index_descriptor_impl<C>
#else
    template<typename C>
    struct reduced_fixed_index_descriptor_impl<C, std::enable_if_t<atomic_fixed_index_descriptor<C>>>
#endif
    {
      using type = Coefficients<C>;
    };


    template<typename C1, typename...Cs>
    struct reduced_fixed_index_descriptor_impl<Coefficients<C1, Cs...>>
    {
      using type = Concatenate<
        typename reduced_fixed_index_descriptor_impl<std::decay_t<C1>>::type,
        typename reduced_fixed_index_descriptor_impl<std::decay_t<Cs>>::type...>;
    };

  } // namespace detail


  /**
   * \brief Reduce a \ref typed_index_descriptor into its canonical form.
   * \sa reduced_typed_index_descriptor_t
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor T>
  struct reduced_fixed_index_descriptor
#else
  template<typename T, typename Enable = void>
  struct reduced_fixed_index_descriptor {};

  template<typename T>
  struct reduced_fixed_index_descriptor<T, std::enable_if_t<fixed_index_descriptor<T>>>
#endif
  {
    using type = typename detail::reduced_fixed_index_descriptor_impl<std::decay_t<T>>::type;
  };


  /**
   * \brief Helper template for \ref reduced_typed_index_descriptor.
   */
  template<typename T>
  using reduced_fixed_index_descriptor_t = typename reduced_fixed_index_descriptor<T>::type;


  // ----------------- //
  //   equivalent_to   //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct is_equivalent_to : std::false_type {};

    template<typename T, typename U>
    struct is_equivalent_to<T, U, std::enable_if_t<
      std::is_same<typename reduced_fixed_index_descriptor<T>::type, typename reduced_fixed_index_descriptor<U>::type>::value>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T is equivalent to U, where T and U are sets of coefficients.
   * \details Sets of coefficients are equivalent if they are treated functionally the same.
   * - Any coefficient or group of coefficients is equivalent to itself.
   * - Coefficient<Ts...> is equivalent to Coefficient<Us...>, if each Ts is equivalent to its respective Us.
   * - Coefficient<T> is equivalent to T, and vice versa.
   * \par Example:
   * <code>equivalent_to&lt;Axis, Coefficients&lt;Axis&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept equivalent_to = fixed_index_descriptor<T> and fixed_index_descriptor<U> and
      std::same_as<reduced_fixed_index_descriptor_t<T>, reduced_fixed_index_descriptor_t<U>>;
#else
  constexpr bool equivalent_to = detail::is_equivalent_to<T, U>::value;
#endif


  // ------------- //
  //   prefix_of   //
  // ------------- //

  namespace detail
  {
    /**
     * \internal
     * \brief Type trait testing whether T (a set of coefficients) is a prefix of U.
     * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
     */
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_prefix_of : std::false_type {};


#ifdef __cpp_concepts
    template<typename C1, typename C2> requires equivalent_to<C1, C2>
    struct is_prefix_of<C1, C2>
#else
    template<typename C1, typename C2>
    struct is_prefix_of<C1, C2, std::enable_if_t<equivalent_to<C1, C2>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typename C>
    struct is_prefix_of<Coefficients<>, C>
#else
    template<typename C>
    struct is_prefix_of<Coefficients<>, C, std::enable_if_t<not equivalent_to<Coefficients<>, C>>>
#endif
      : std::true_type {};


    template<typename C1, typename...Cs>
    struct is_prefix_of<C1, Coefficients<C1, Cs...>> : std::true_type {};


#ifdef __cpp_concepts
    template<typename C, typename...C1, typename...C2>
    struct is_prefix_of<Coefficients<C, C1...>, Coefficients<C, C2...>>
#else
    template<typename C, typename...C1, typename...C2>
    struct is_prefix_of<Coefficients<C, C1...>, Coefficients<C, C2...>, std::enable_if_t<
      (not equivalent_to<Coefficients<C, C1...>, Coefficients<C, C2...>>)>>
#endif
      : std::bool_constant<is_prefix_of<Coefficients<C1...>, Coefficients<C2...>>::value> {};

  } // namespace detail


  /**
   * \brief T is a prefix of U, where T and U are sets of coefficients.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of Coefficients<C, Cs...> for any typed index descriptors Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * Coefficients<> is a prefix of any set of coefficients.
   * \par Example:
   * <code>prefix_of&lt;Coefficients&lt;Axis&gt;, Coefficients&lt;Axis, angle::Radians&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of =
#else
  constexpr bool prefix_of =
#endif
    index_descriptor<T> and index_descriptor<U> and detail::is_prefix_of<
      reduced_fixed_index_descriptor_t<T>, reduced_fixed_index_descriptor_t<U>>::value;


  // --------------------------------------------------------- //
  //   has_uniform_dimension_type, uniform_dimension_type_of   //
  // --------------------------------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename C>
#else
    template<typename C, typename = void>
#endif
    struct uniform_dimension_impl : std::false_type {};


#ifdef __cpp_concepts
    template<atomic_fixed_index_descriptor C> requires (dimension_size_of_v<C> == 1)
    struct uniform_dimension_impl<C>
#else
    template<typename C>
    struct uniform_dimension_impl<C, std::enable_if_t<atomic_fixed_index_descriptor<C> and (dimension_size_of_v<C> == 1)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<atomic_fixed_index_descriptor C> requires (dimension_size_of_v<C> == 1)
    struct uniform_dimension_impl<Coefficients<C>>
#else
    template<typename C>
    struct uniform_dimension_impl<Coefficients<C>, std::enable_if_t<
      atomic_fixed_index_descriptor<C> and (dimension_size_of_v<C> == 1)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<atomic_fixed_index_descriptor C, fixed_index_descriptor...Cs> requires (dimension_size_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and std::same_as<C, typename uniform_dimension_impl<Coefficients<Cs...>>::uniform_type>
    struct uniform_dimension_impl<Coefficients<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct uniform_dimension_impl<Coefficients<C, Cs...>, std::enable_if_t<
      atomic_fixed_index_descriptor<C> and (... and fixed_index_descriptor<Cs>) and (dimension_size_of_v<C> == 1) and
        (sizeof...(Cs) > 0) and std::is_same<C, typename uniform_dimension_impl<Coefficients<Cs...>>::uniform_type>::value>>
#endif
      : std::true_type { using uniform_type = C; };


#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct has_uniform_dimension_type_impl : std::false_type {};

    template<typename T>
    struct has_uniform_dimension_type_impl<T, std::enable_if_t<
      detail::uniform_dimension_impl<typename reduced_fixed_index_descriptor<T>::type>::value>> : std::true_type {};
#endif

  } // namespace detail


  /**
   * \brief T is an fixed-type index descriptor comprising a uniform set of 1D \ref atomic_fixed_index_descriptor types.
   * \tparam T
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_uniform_dimension_type = fixed_index_descriptor<T> and
    ((untyped_index_descriptor<T> and dimension_size_of_v<T> >= 1) or
      detail::uniform_dimension_impl<reduced_fixed_index_descriptor_t<T>>::value);
#else
  constexpr bool has_uniform_dimension_type = detail::has_uniform_dimension_type_impl<T>::value;
#endif


  /**
   * \brief If T \ref has_uniform_dimension_type, member <code>type</code> is an alias for that type.
   * \sa uniform_dimension_type_of_t
   */
#ifdef __cpp_concepts
  template<has_uniform_dimension_type T>
  struct uniform_dimension_type_of
#else
  template<typename T, typename Enable = void>
  struct uniform_dimension_type_of {};

  template<typename T>
  struct uniform_dimension_type_of<T, std::enable_if_t<detail::has_uniform_dimension_type_impl<T>::value>>
#endif
  {
    using type = typename detail::uniform_dimension_impl<reduced_fixed_index_descriptor_t<T>>::uniform_type;
  };


  /**
   * \brief Helper template for \ref uniform_dimension_type_of.
   */
#ifdef __cpp_concepts
  template<has_uniform_dimension_type T>
#else
  template<typename T>
#endif
  using uniform_dimension_type_of_t = typename uniform_dimension_type_of<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
