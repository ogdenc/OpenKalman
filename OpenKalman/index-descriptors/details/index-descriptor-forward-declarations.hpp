/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward definitions for coefficient types.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP

#include <type_traits>
#include <functional>

namespace OpenKalman
{
  // ---------------------------- //
  //   atomic_coefficient_group   //
  // ---------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_atomic_coefficient_group : std::false_type {};
  }


  /**
   * \brief T is an atomic (non-seperable) group of coefficients.
   * \details Atomic coefficient groups are %coefficients or groups of %coefficients that function as a unit,
   * and cannot be separated. They may be combined into composite coefficients by passing them as template
   * parameters to Coefficients.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept atomic_coefficient_group =
#else
  constexpr bool atomic_coefficient_group =
#endif
    detail::is_atomic_coefficient_group<std::decay_t<T>>::value;


  // -------------------------- //
  //   composite_coefficients   //
  // -------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_composite_coefficients : std::false_type {};
  }


  /**
   * \brief T is a composite set of coefficient groups.
   * \details Composite coefficients are instances of the class Coefficients, which has the purpose of grouping
   * other atomic or composite coefficients. Composite coefficients can, themselves, comprise groups of other
   * composite components.
   * \sa Coefficients.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept composite_coefficients = detail::is_composite_coefficients<std::decay_t<T>>::value;
#else
  constexpr bool composite_coefficients = detail::is_composite_coefficients<std::decay_t<T>>::value;
#endif


  // ---------------------- //
  //   fixed_coefficients   //
  // ---------------------- //

  /**
   * \brief T is a fixed (defined at compile time) set of coefficients.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept fixed_coefficients = composite_coefficients<std::decay_t<T>> or atomic_coefficient_group<std::decay_t<T>>;
#else
  template<typename T>
  constexpr bool fixed_coefficients = composite_coefficients<std::decay_t<T>> or
    atomic_coefficient_group<std::decay_t<T>>;
#endif


  // ----------------------------- //
  //   Atomic coefficient groups   //
  // ----------------------------- //

  // Documentation in Axis.hpp
  struct Axis;


  // Documentation in Distance.hpp
  struct Distance;


  // Documentation in Angle.hpp
  template<template<typename Scalar> typename Limits>
#ifdef __cpp_concepts
  requires std::floating_point<decltype(Limits<double>::min)> and
    std::floating_point<decltype(Limits<double>::max)> and (Limits<double>::min < Limits<double>::max)
#endif
  struct Angle;


  // Documentation in Inclination.hpp
  template<template<typename Scalar> typename Limits>
#ifdef __cpp_concepts
  requires std::floating_point<decltype(Limits<double>::down)> and
    std::floating_point<decltype(Limits<double>::up)> and (Limits<double>::down < Limits<double>::up)
#endif
  struct Inclination;


  // Documentation in Polar.hpp
#ifdef __cpp_concepts
  template<fixed_coefficients C1, fixed_coefficients C2>
#else
  template<typename C1, typename C2, typename = void>
#endif
  struct Polar;


  // Documentation in Spherical.hpp
#ifdef __cpp_concepts
  template<fixed_coefficients C1, fixed_coefficients C2, fixed_coefficients C3>
#else
  template<typename C1, typename C2, typename C3, typename = void>
#endif
  struct Spherical;


  namespace detail
  {
    template<>
    struct is_atomic_coefficient_group<Axis> : std::true_type {};

    template<>
    struct is_atomic_coefficient_group<Distance> : std::true_type {};

    template<template<typename Scalar> typename Limits>
    struct is_atomic_coefficient_group<Angle<Limits>> : std::true_type {};

    template<template<typename Scalar> typename Limits>
    struct is_atomic_coefficient_group<Inclination<Limits>> : std::true_type {};

    template<typename C1, typename C2>
    struct is_atomic_coefficient_group<Polar<C1, C2>> : std::true_type {};

    template<typename C1, typename C2, typename C3>
    struct is_atomic_coefficient_group<Spherical<C1, C2, C3>> : std::true_type {};

  } // namespace detail


  // ---------------- //
  //   Coefficients   //
  // ---------------- //

  // Documentation in Coefficients.hpp
#ifdef __cpp_concepts
  template<fixed_coefficients...Cs>
#else
  template<typename...Cs>
#endif
  struct Coefficients;


  namespace detail
  {
    template<typename...C>
    struct is_composite_coefficients<Coefficients<C...>> : std::true_type {};
  }


  // ----------------------- //
  //   DynamicCoefficients   //
  // ----------------------- //

  // Documentation in DynamicCoefficients.hpp
  template<typename Scalar>
  struct DynamicCoefficients;


  // ------------------------ //
  //   dynamic_coefficients   //
  // ------------------------ //

  namespace detail
  {
    template<typename T>
    struct is_dynamic_coefficients : std::false_type {};

    template<typename Scalar>
    struct is_dynamic_coefficients<DynamicCoefficients<Scalar>> : std::true_type {};
  }


  /**
   * \brief T is a dynamic (defined at run time) set of coefficients.
   * \sa DynamicCoefficients.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept dynamic_coefficients = detail::is_dynamic_coefficients<std::decay_t<T>>::value;
#else
  template<typename T>
  constexpr bool dynamic_coefficients = detail::is_dynamic_coefficients<std::decay_t<T>>::value;
#endif


  // -------------- //
  //   Dimensions   //
  // -------------- //

  // Documentation in Dimensions.hpp
  template<std::size_t size>
  struct Dimensions;


  namespace detail
  {
    template<std::size_t size>
    struct is_composite_coefficients<Dimensions<size>> : std::bool_constant<size != dynamic_size> {};
  }


  // ---------------------------- //
  //   untyped_index_descriptor   //
  // ---------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_untyped_index_descriptor : std::false_type {};

    template<std::size_t size>
    struct is_untyped_index_descriptor<Dimensions<size>> : std::true_type {};

    template<>
    struct is_untyped_index_descriptor<Axis> : std::true_type {};

    template<typename...Cs>
    struct is_untyped_index_descriptor<Coefficients<Cs...>>
      : std::bool_constant<(is_untyped_index_descriptor<Cs>::value and ...)> {};
  }


  /**
   * \brief A descriptor for a tensor index.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept untyped_index_descriptor =
#else
  constexpr bool untyped_index_descriptor =
#endif
    detail::is_untyped_index_descriptor<std::decay_t<T>>::value or std::is_integral_v<std::decay_t<T>>;


  // -------------------------- //
  //   typed_index_descriptor   //
  // -------------------------- //

  /**
   * \brief T is a group of atomic or composite coefficients, or dynamic coefficients.
   * \details Atomic coefficient groups are %coefficients or groups of %coefficients that function as a unit,
   * and cannot be separated. They may be combined into composite coefficients by passing them as template
   * parameters to Coefficients. These include Axis, Distance, Angle, Inclination, Polar, and Spherical.
   *
   * Composite coefficients are specializations of the class Coefficients, which has the purpose of grouping
   * other atomic or composite coefficients. Composite coefficients can, themselves, comprise groups of other
   * composite components. Composite coefficients are of the form Coefficients<Cs...>.
   *
   * Dynamic coefficients are defined at runtime.
   * <b>Examples</b>:
   * - Axis
   * - Polar<Distance, angle::Radians>
   * - Coefficients<Axis, angle::Radians>
   * - Coefficients<Spherical<angle::Degrees, inclination::degrees, Distance>, Axis, Axis>
   * - DynamicCoefficients
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_index_descriptor =
#else
  constexpr bool typed_index_descriptor =
#endif
    (not untyped_index_descriptor<T>) and (fixed_coefficients<T> or dynamic_coefficients<T>);


  // -------------------- //
  //   index_descriptor   //
  // -------------------- //

  /**
   * \brief A descriptor for a tensor index.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept index_descriptor =
#else
  constexpr bool index_descriptor =
#endif
    untyped_index_descriptor<T> or typed_index_descriptor<T>;


  // --------------------- //
  //   dimension_size_of   //
  // --------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_size_of;


#ifdef __cpp_concepts
  template<index_descriptor T> requires std::is_integral_v<std::decay_t<T>>
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<index_descriptor<T> and std::is_integral_v<std::decay_t<T>>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<index_descriptor T> requires (not std::is_integral_v<std::decay_t<T>>)
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<index_descriptor<T> and not std::is_integral_v<std::decay_t<T>>>>
#endif
    : std::integral_constant<std::size_t, std::decay_t<T>::value> {};


  /**
   * \brief Helper template for \ref dimension_size_of.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T>
#endif
  constexpr auto dimension_size_of_v = dimension_size_of<std::decay_t<T>>::value;


  // ------------------------------- //
  //   euclidean_dimension_size_of   //
  // ------------------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor if it is transformed into Euclidean space.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, typename Enable = void>
#endif
  struct euclidean_dimension_size_of;


#ifdef __cpp_concepts
  template<untyped_index_descriptor T>
  struct euclidean_dimension_size_of<T>
#else
  template<typename T>
  struct euclidean_dimension_size_of<T, std::enable_if_t<untyped_index_descriptor<T>>>
#endif
    : dimension_size_of<T> {};


  // Note: euclidean_dimension_size_of for typed_index_descriptor types are defined elsewhere.


  /**
   * \brief Helper template for \ref euclidean_dimension_size_of.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T>
#endif
  constexpr auto euclidean_dimension_size_of_v = euclidean_dimension_size_of<std::decay_t<T>>::value;


  // --------------------------- //
  //   dimension_difference_of   //
  // --------------------------- //

  /**
   * \brief The type of the \ref index_descriptor when tensors having respective index_descriptors T are subtracted.
   * \details For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Axis, so if
   * <code>T</code> is Distance, the resulting <code>type</code> will be Axis.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_difference_of;


#ifdef __cpp_concepts
  template<untyped_index_descriptor T>
  struct dimension_difference_of<T>
#else
  template<typename T>
  struct dimension_difference_of<T, std::enable_if_t<untyped_index_descriptor<T>>>
#endif
  {
    using type = std::decay_t<T>;
  };


  // Note: dimension_difference_of for typed_index_descriptor types are defined elsewhere.


  /**
   * \brief Helper template for \ref dimension_difference_of.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T>
#endif
  using dimension_difference_of_t = typename dimension_difference_of<std::decay_t<T>>::type;


  // ------------------------- //
  //   get_dimension_size_of   //
  // ------------------------- //

  /**
   * \brief Get the dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_dimension_size_of(const T& t)
  {
    return t;
  }


  // ----------------------------------- //
  //   get_euclidean_dimension_size_of   //
  // ----------------------------------- //

  /**
   * \brief Get the Euclidean dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_euclidean_dimension_size_of(const T& t)
  {
    if constexpr (untyped_index_descriptor<T>)
      return get_dimension_size_of(t);
    else if constexpr (dynamic_coefficients<T>)
      return T::runtime_euclidean_dimension;
    else
      return euclidean_dimension_size_of_v<T>;
  }


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
  template<fixed_coefficients C, std::size_t N> requires (N != dynamic_size)
#else
  template<typename C, std::size_t N, std::enable_if_t<fixed_coefficients<C> and N != dynamic_size, int> = 0>
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
  template<fixed_coefficients ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
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
      using type = replicated_fixed_index_descriptor<Axis, N>;
    };


    template<typename C>
    struct reduced_fixed_index_descriptor_impl<Coefficients<C>>
    {
      using type = typename reduced_fixed_index_descriptor_impl<std::decay_t<C>>::type;
    };


    template<typename C1, typename...Cs>
    struct reduced_fixed_index_descriptor_impl<Coefficients<C1, Cs...>>
    {
      using type = Concatenate<
        typename reduced_fixed_index_descriptor_impl<std::decay_t<C1>>::type,
        typename reduced_fixed_index_descriptor_impl<std::decay_t<Cs>>::type...>;
    };


#ifdef __cpp_concepts
    template<template<typename...> typename T, typename...Cs> requires atomic_coefficient_group<T<Cs...>>
    struct reduced_fixed_index_descriptor_impl<T<Cs...>>
#else
    template<template<typename...> typename T, typename...Cs>
    struct reduced_fixed_index_descriptor_impl<T<Cs...>, std::enable_if_t<atomic_coefficient_group<T<Cs...>>>>
#endif
    {
      using type = T<typename reduced_fixed_index_descriptor_impl<std::decay_t<Cs>>::type...>;
    };

  } // namespace detail


  /**
   * \brief Reduce a \ref typed_index_descriptor into its canonical form.
   * \sa reduced_typed_index_descriptor_t
   */
#ifdef __cpp_concepts
  template<fixed_coefficients T>
  struct reduced_fixed_index_descriptor
#else
  template<typename T, typename Enable = void>
  struct reduced_fixed_index_descriptor {};

  template<typename T>
  struct reduced_fixed_index_descriptor<T, std::enable_if_t<fixed_coefficients<T>>>
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
  concept equivalent_to = index_descriptor<T> and index_descriptor<U> and
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
    template<atomic_coefficient_group C> requires (dimension_size_of_v<C> == 1)
    struct uniform_dimension_impl<C>
#else
    template<typename C>
    struct uniform_dimension_impl<C, std::enable_if_t<atomic_coefficient_group<C> and (dimension_size_of_v<C> == 1)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<atomic_coefficient_group C> requires (dimension_size_of_v<C> == 1)
    struct uniform_dimension_impl<Coefficients<C>>
#else
    template<typename C>
    struct uniform_dimension_impl<Coefficients<C>, std::enable_if_t<
      atomic_coefficient_group<C> and (dimension_size_of_v<C> == 1)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<atomic_coefficient_group C, fixed_coefficients...Cs> requires (dimension_size_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and std::same_as<C, typename uniform_dimension_impl<Coefficients<Cs...>>::uniform_type>
    struct uniform_dimension_impl<Coefficients<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct uniform_dimension_impl<Coefficients<C, Cs...>, std::enable_if_t<
      atomic_coefficient_group<C> and (... and fixed_coefficients<Cs>) and (dimension_size_of_v<C> == 1) and
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
   * \brief T is an fixed-type index descriptor comprising a uniform set of 1D \ref atomic_coefficient_group types.
   * \tparam T
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_uniform_dimension_type = fixed_coefficients<T> and
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


  // -------------- //
  //   Arithmetic   //
  // -------------- //

#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  /**
   * \brief Three-way comparison for \ref fixed_coefficients.
   * \sa \ref equivalent_to
   */
  template<fixed_coefficients A, fixed_coefficients B>
  constexpr auto operator<=>(A&& a, B&& b)
  {
    if constexpr (untyped_index_descriptor<A> and untyped_index_descriptor<B>)
      return dimension_size_of_v<A> <=> dimension_size_of_v<B>;
    else if constexpr (equivalent_to<A, B>) return std::partial_ordering::equivalent;
    else return std::partial_ordering::unordered;
  }
#else
  /**
   * \brief Compares for equivalence.
   * \sa \ref equivalent_to
   */
  template<typename A, typename B, std::enable_if_t<fixed_coefficients<A> and fixed_coefficients<B>, int> = 0>
  constexpr bool operator==(A&& a, B&& b)
  {
    return equivalent_to<A, B>;
  }


  /**
   * \brief Compares for non-equivalence.
   * \sa \ref equivalent_to
   */
  template<typename A, typename B, std::enable_if_t<fixed_coefficients<A> and fixed_coefficients<B>, int> = 0>
  constexpr bool operator!=(A&& a, B&& b)
  {
    return not equivalent_to<A, B>;
  }
#endif


  /**
   * \brief Add two \ref index_descriptor values, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<index_descriptor T, index_descriptor U> requires
    (not (typed_index_descriptor<T> or typed_index_descriptor<U>) or (fixed_coefficients<T> and fixed_coefficients<U>))
#else
  template<typename T, typename U, std::enable_if_t<index_descriptor<T> and index_descriptor<U> and
    (not (typed_index_descriptor<T> or typed_index_descriptor<U>) or (fixed_coefficients<T> and fixed_coefficients<U>)), int> = 0>
#endif
  constexpr auto operator+(const T& t, const U& u) noexcept
  {
    if constexpr (typed_index_descriptor<T> or typed_index_descriptor<U>)
    {
      return Concatenate<T, U> {};
    }
    else
    {
      if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
        return Dimensions{t() + u()};
      else
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
    }
  }


  /**
   * \brief Subtract two \ref untyped_index_descriptor values, whether fixed or dynamic.
   * \warning This does not perform any runtime checks to ensure that the result is non-negative.
   */
#ifdef __cpp_concepts
  template<untyped_index_descriptor T, untyped_index_descriptor U> requires (dimension_size_of_v<T> == dynamic_size) or
    (dimension_size_of_v<U> == dynamic_size) or (dimension_size_of_v<T> > dimension_size_of_v<U>)
#else
  template<typename T, typename U, std::enable_if_t<untyped_index_descriptor<T> and untyped_index_descriptor<U> and
    ((dimension_size_of<T>::value == dynamic_size) or (dimension_size_of<U>::value == dynamic_size) or
      (dimension_size_of<T>::value > dimension_size_of<U>::value)), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u) noexcept
  {
    if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
      return Dimensions{t() - u()};
    else
      return Dimensions<dimension_size_of_v<T> - dimension_size_of_v<U>>{};
  }


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
