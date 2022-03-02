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
 * \brief Forward declarations for traits relating to OpenKalman or native matrix types.
 */

#ifndef OPENKALMAN_FORWARD_TRAITS_HPP
#define OPENKALMAN_FORWARD_TRAITS_HPP

#include <type_traits>
#include <complex>


/**
 * \namespace OpenKalman
 * \brief The root namespace for OpenKalman.
 *
 * \internal
 * \namespace OpenKalman::internal
 * \brief Namespace for internal definitions, not intended for use outside of OpenKalman development.
 */


namespace OpenKalman
{

  // ---------------- //
  //   Coefficients   //
  // ---------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether T is an atomic group of coefficients.
     * \details Atomic coefficient groups are %coefficients or groups of %coefficients that function as a unit,
     * and cannot be separated. They may be combined into composite coefficients by passing them as template
     * parameters to Coefficients.
     */
    template<typename T>
    struct is_atomic_coefficient_group : std::false_type {};


    /**
     * \internal
     * \brief A type trait testing whether T is a composite set of coefficient groups.
     * \details Composite coefficients are specializations of the class Coefficients, which has the purpose of grouping
     * other atomic or composite coefficients. Composite coefficients can, themselves, comprise groups of other
     * composite components.
     */
    template<typename T>
    struct is_composite_coefficients : std::false_type {};


    /**
     * \internal
     * \brief A type trait testing whether T is a dynamic (defined at time) set of coefficients.
     * \sa DynamicCoefficients.
     */
    template<typename T>
    struct is_dynamic_coefficients : std::false_type {};
  }


  /**
   * \brief T is an atomic (non-seperable) group of coefficients.
   * \details Atomic coefficient groups are %coefficients or groups of %coefficients that function as a unit,
   * and cannot be separated. They may be combined into composite coefficients by passing them as template
   * parameters to Coefficients.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept atomic_coefficient_group = internal::is_atomic_coefficient_group<std::decay_t<T>>::value;
#else
  constexpr bool atomic_coefficient_group = internal::is_atomic_coefficient_group<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is a composite set of coefficient groups.
   * \details Composite coefficients are specializations of the class Coefficients, which has the purpose of grouping
   * other atomic or composite coefficients. Composite coefficients can, themselves, comprise groups of other
   * composite components.
   * \sa Coefficients.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept composite_coefficients = internal::is_composite_coefficients<std::decay_t<T>>::value;
#else
  constexpr bool composite_coefficients = internal::is_composite_coefficients<std::decay_t<T>>::value;
#endif


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


  /**
   * \brief T is a dynamic (defined at run time) set of coefficients.
   * \sa DynamicCoefficients.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept dynamic_coefficients = internal::is_dynamic_coefficients<std::decay_t<T>>::value;
#else
  template<typename T>
  constexpr bool dynamic_coefficients = internal::is_dynamic_coefficients<std::decay_t<T>>::value;
#endif


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
  concept coefficients = fixed_coefficients<T> or dynamic_coefficients<T>;
#else
  constexpr bool coefficients = fixed_coefficients<T> or dynamic_coefficients<T>;
#endif


  namespace internal
  {
    /**
     * \internal
     * \brief Type trait testing whether coefficients T are equivalent to coefficients U.
     * \details Sets of coefficients are equivalent if they are treated functionally the same.
     */
#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_equivalent_to : std::false_type {};
  }


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
  concept equivalent_to = internal::is_equivalent_to<T, U>::value;
#else
  constexpr bool equivalent_to = internal::is_equivalent_to<T, U>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * \brief Type trait testing whether T (a set of coefficients) is a prefix of U.
     * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
     */
#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_prefix_of : std::false_type {};
  } // namespace internal


  /**
   * \brief T is a prefix of U, where T and U are sets of coefficients.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of Coefficients<C, Cs...> for any coefficients Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * Coefficients<> is a prefix of any set of coefficients.
   * \par Example:
   * <code>prefix_of&lt;Coefficients&lt;Axis&gt;, Coefficients&lt;Axis, angle::Radians&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of = internal::is_prefix_of<T, U>::value;
#else
  constexpr bool prefix_of = internal::is_prefix_of<T, U>::value;
#endif


  // -------------------- //
  //    complex_number    //
  // -------------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_complex_number : std::false_type {};


    template<typename T>
    struct is_complex_number<std::complex<T>> : std::true_type {};
  }


  /**
   * \brief T is a std::complex.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept complex_number = internal::is_complex_number<std::decay_t<T>>::value;
#else
  constexpr bool complex_number = internal::is_complex_number<std::decay_t<T>>::value;
#endif


  // ----------------- //
  //    scalar_type    //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_scalar_type : std::false_type {};


    template<typename T>
    struct is_scalar_type<T, std::enable_if_t<
      std::is_convertible_v<decltype(std::declval<T>() + std::declval<T>()), const std::decay_t<T>> and
      std::is_convertible_v<decltype(std::declval<T>() - std::declval<T>()), const std::decay_t<T>> and
      std::is_convertible_v<decltype(std::declval<T>() * std::declval<T>()), const std::decay_t<T>> and
      std::is_convertible_v<decltype(std::declval<T>() / std::declval<T>()), const std::decay_t<T>>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T is a scalar type (i.e., is arithmetic, complex, or other number in which + - * and / are defined).
   * \details OpenKalman presumes that elements of an algebraic field may be the entries of a matrix.
   * This includes integral, floating point, and complex values. T need not be an algebraic field, but this concept
   * is designed, conceptually, to capture the idea of a non-commutative division ring that is not necessarily
   * associative under multiplication.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept scalar_type = std::is_arithmetic_v<T> or complex_number<T> or
    requires(T t1, T t2) {
      {t1 + t2} -> std::convertible_to<const std::decay_t<T>>;
      {t1 - t2} -> std::convertible_to<const std::decay_t<T>>;
      {t1 * t2} -> std::convertible_to<const std::decay_t<T>>;
      {t1 / t2} -> std::convertible_to<const std::decay_t<T>>; };
#else
  constexpr bool scalar_type = std::is_arithmetic_v<T> or complex_number<T> or
    detail::is_scalar_type<std::decay_t<T>>::value;
#endif


  // ---------------- //
  //  scalar_type_of  //
  // ---------------- //

  /**
   * \brief Type scalar type (e.g., std::float, std::double, std::complex<double>) of a matrix, expression, or array.
   * \tparam T A matrix, expression, or array.
   * \internal \sa interface::ScalarTypeOf
   */
#ifdef __cpp_concepts
  template<typename T> requires requires { typename interface::ScalarTypeOf<std::decay_t<T>>; }
#else
  template<typename T>
#endif
  using scalar_type_of = typename interface::ScalarTypeOf<std::decay_t<T>>;


  /**
   * \brief helper template for \ref scalar_type_of.
   */
#ifdef __cpp_concepts
  template<typename T> requires requires { typename interface::ScalarTypeOf<std::decay_t<T>>::type; }
#else
  template<typename T>
#endif
  using scalar_type_of_t = typename scalar_type_of<T>::type;


  // ---------------- //
  //  max_indices_of  //
  // ---------------- //

  /**
   * \brief The maximum number of indices of structure T.
   * \tparam T A tensor (vector, matrix, etc.)
   */
  template<typename T>
  struct max_indices_of
    : std::integral_constant<std::size_t, interface::StorageArrayTraits<std::decay_t<T>>::max_indices> {};


  /**
   * \brief helper template for \ref max_indices_of.
   */
  template<typename T>
  static constexpr auto max_indices_of_v = max_indices_of<T>::value;


  // ----------- //
  //  indexible  //
  // ----------- //

#ifndef __cpp_lib_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_indexible : std::false_type {};

    template<typename T>
    struct is_indexible<T, std::enable_if_t<(max_indices_of<T>::value > 0)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T is a generalized tensor type.
   * \details T can be a tensor over a vector space, but can also be an analogous algebraic structure over a
   * tensor product of modules over division rings (e.g., an vector-like structure that contains angles).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept indexible = (max_indices_of_v<T> > 0);
#else
  constexpr bool indexible = detail::is_indexible<T>::value;
#endif


  // -------------------- //
  //  index_dimension_of  //
  // -------------------- //

  /**
   * \brief The dimension of an index for a matrix, expression, or array.
   * \note If the dimension is dynamic, then <code>value</code> is \ref dynamic_size.
   * \tparam N The index
   * \tparam T The matrix, expression, or array
   * \internal \sa interface::IndexTraits
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N = 0>
#else
  template<typename T, std::size_t N = 0, typename = void>
#endif
  struct index_dimension_of : std::integral_constant<std::size_t, 0> {};


#ifdef __cpp_concepts
  template<typename T, std::size_t N> requires (N < max_indices_of_v<T>)
  struct index_dimension_of<T, N>
#else
  template<typename T, std::size_t N>
  struct index_dimension_of<T, N, std::enable_if_t<N < max_indices_of<T>::value>>
#endif
    : std::integral_constant<std::size_t, interface::IndexTraits<std::decay_t<T>, N>::dimension> {};


  /**
   * \brief helper template for \ref index_dimension_of.
   */
  template<typename T, std::size_t N = 0>
  static constexpr auto index_dimension_of_v = index_dimension_of<T, N>::value;


  // ------------------ //
  //  row_dimension_of  //
  // ------------------ //

  /**
   * \brief The row dimension of a matrix, expression, or array.
   * \note If the row dimension is dynamic, then <code>value</code> is \ref dynamic_size.
   * \tparam T The matrix, expression, or array.
   * \internal \sa interface::IndexTraits
   */
  template<typename T>
  using row_dimension_of = index_dimension_of<T, 0>;


  /**
   * \brief helper template for \ref row_dimension_of.
   */
  template<typename T>
  static constexpr auto row_dimension_of_v = row_dimension_of<T>::value;


  // --------------------- //
  //  column_dimension_of  //
  // --------------------- //

  /**
   * \brief The column dimension of a matrix, expression, or array.
   * \note If the column dimension is dynamic, then <code>value</code> is \ref dynamic_size.
   * \tparam T The matrix, expression, or array.
   * \internal \sa interface::IndexTraits
   */
  template<typename T>
  using column_dimension_of = index_dimension_of<T, 1>;


  /**
   * \brief helper template for \ref column_dimension_of.
   */
  template<typename T>
  static constexpr auto column_dimension_of_v = column_dimension_of<T>::value;


  // ------------------- //
  //  dynamic_dimension  //
  // ------------------- //

  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept dynamic_dimension =
#else
  constexpr bool dynamic_dimension =
#endif
    (N < max_indices_of_v<T>) and (index_dimension_of_v<T, N> == dynamic_size);


  // -------------- //
  //  dynamic_rows  //
  // -------------- //

  /**
   * \brief Specifies that T has a row dimension that is defined at run time.
   * \details The matrix library interface will specify this for native matrices and expressions.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_rows =
#else
  constexpr bool dynamic_rows =
#endif
    dynamic_dimension<T, 0>;


  // ----------------- //
  //  dynamic_columns  //
  // ----------------- //

  /**
   * \brief Specifies that T has a column dimension that is defined at run time.
   * \details The matrix library interface will specify this for native matrices and expressions.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_columns =
#else
  constexpr bool dynamic_columns =
#endif
    dynamic_dimension<T, 1>;


  // ----------------------- //
  //  any_dynamic_dimension  //
  // ----------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool any_dynamic_dimension_impl(std::index_sequence<I...>) { return (... or dynamic_dimension<T, I>); }
  }


  /**
   * \brief Specifies that T has a dynamic dimension for at least one of its indices.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept any_dynamic_dimension =
#else
  constexpr bool any_dynamic_dimension =
#endif
    detail::any_dynamic_dimension_impl<T>(std::make_index_sequence<max_indices_of_v<T>> {});


  // ------------------ //
  //  element_gettable  //
  // ------------------ //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void, typename...I>
    struct element_gettable_impl : std::false_type {};

    template<typename T, typename...I>
    struct element_gettable_impl<T, std::enable_if_t<(std::is_convertible_v<I, const std::size_t&> and ...) and
      std::is_void<std::void_t<decltype(interface::GetElement<std::decay_t<T>, void, I...>::get(
        std::declval<T&&>(), std::declval<I>()...))>>::value and (sizeof...(I) > 0)>, I...>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type has elements that can be retrieved with indices I... (of type std::size_t).
   */
  template<typename T, typename...I>
#ifdef __cpp_concepts
  concept element_gettable = (std::convertible_to<I, const std::size_t&> and ...) and (sizeof...(I) > 0) and
    requires(T&& t, I...i) { interface::GetElement<std::decay_t<T>, I...>::get(std::forward<T>(t), i...); };
#else
  constexpr bool element_gettable = detail::element_gettable_impl<T, void, I...>::value;
#endif


  // ------------------ //
  //  element_settable  //
  // ------------------ //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void, typename...I>
    struct element_settable_impl : std::false_type {};

    template<typename T, typename...I>
    struct element_settable_impl<T, std::enable_if_t<(std::is_convertible_v<I, const std::size_t&> and ...) and
      (not std::is_const_v<std::remove_reference_t<T>>) and
      std::is_void<std::void_t<decltype(interface::SetElement<std::decay_t<T>, void, I...>::set(
        std::declval<std::remove_reference_t<T>&>(),
        std::declval<const scalar_type_of_t<T>&>(), std::declval<I>()...))>>::value and (sizeof...(I) > 0)>, I...>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type has elements that can be set with indices I... (of type std::size_t).
   */
  template<typename T, typename...I>
#ifdef __cpp_concepts
  concept element_settable = (std::convertible_to<I, const std::size_t&> and ...) and
    (not std::is_const_v<std::remove_reference_t<T>>) and
    requires(std::remove_reference_t<T>& t, const scalar_type_of_t<T>& s, I...i) {
      interface::SetElement<std::decay_t<T>, I...>::set(t, s, i...);
    };
#else
  constexpr bool element_settable = detail::element_settable_impl<T, void, I...>::value;
#endif


  // --------------------------------------- //
  //  has_nested_matrix, nested_matrix_of_t  //
  // --------------------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_nested_matrix_impl : std::false_type {};

    template<typename T>
    struct has_nested_matrix_impl<T, std::enable_if_t<
      (std::tuple_size<typename interface::Dependencies<T>::type>::value > 0)>> : std::true_type {};
  }
#endif

  /**
   * \brief A matrix that has a nested matrix, if it is a wrapper type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_nested_matrix = (std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type> > 0);
#else
  constexpr bool has_nested_matrix = detail::has_nested_matrix_impl<std::decay_t<T>>::value;
#endif


  /**
   * \brief A wrapper type's nested matrix, if it exists.
   * \details For example, for OpenKalman::Mean<RowCoefficients, M>, the nested matrix type is M.
   * \tparam T A wrapper type that has a nested matrix.
   * \tparam i Index of the dependency (0 by default)
   * \internal \sa interface::Dependencies
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t i = 0> requires
    (i < std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type>)
  using nested_matrix_of = std::tuple_element<i, typename interface::Dependencies<std::decay_t<T>>::type>;
#else
  template<typename T, std::size_t i = 0, typename = void>
  struct nested_matrix_of {};

  template<typename T, std::size_t i>
  struct nested_matrix_of<T, i, std::enable_if_t<
    (i < std::tuple_size<typename interface::Dependencies<std::decay_t<T>>::type>::value)>>
    : std::tuple_element<i, typename interface::Dependencies<std::decay_t<T>>::type> {};
#endif


  /**
   * \brief Helper type for \ref nested_matrix_of.
   * \tparam T A wrapper type that has a nested matrix.
   * \tparam i Index of the dependency (0 by default)
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t i = 0> requires
      (i < std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type>)
  using nested_matrix_of_t = typename nested_matrix_of<T, i>::type;
#else
  template<typename T, std::size_t i = 0>
  using nested_matrix_of_t = typename nested_matrix_of<T, i>::type;
#endif


  // ---------------- //
  //  self_contained  //
  // ---------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct self_contained_impl : std::false_type {};


    template<typename Tup, std::size_t...I>
    constexpr bool no_lvalue_ref_dependencies(std::index_sequence<I...>)
    {
      return (self_contained_impl<std::tuple_element_t<I, Tup>>::value and ...);
    }


#ifdef __cpp_concepts
    template<typename T> requires (not std::is_lvalue_reference_v<T>) and
      (detail::no_lvalue_ref_dependencies<typename interface::Dependencies<std::decay_t<T>>::type>(
        std::make_index_sequence<std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type>> {}))
    struct self_contained_impl<T> : std::true_type {};
#else
    template<typename T>
    struct self_contained_impl<T, std::enable_if_t<
      (std::tuple_size<typename interface::Dependencies<std::decay_t<T>>::type>::value >= 0)>>
      : std::bool_constant<(not std::is_lvalue_reference_v<T>) and
          no_lvalue_ref_dependencies<typename interface::Dependencies<std::decay_t<T>>::type>(
          std::make_index_sequence<std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type>> {})> {};
#endif
  } // namespace detail


  /**
   * \brief Specifies that a type is a self-contained matrix or expression.
   * \details A value is self-contained if it can be created in a function and returned as the result.
   * \sa make_self_contained, equivalent_self_contained_t
   * \internal \sa Dependencies
   */
  template<typename T>
#ifdef __cpp_concepts
  concept self_contained = detail::self_contained_impl<T>::value;
#else
  constexpr bool self_contained = detail::self_contained_impl<T>::value;
#endif


  // --------------------------------------- //
  //  constant_matrix, constant_coefficient  //
  // --------------------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_constant_matrix : std::false_type {};

    template<typename T>
    struct is_constant_matrix<T, std::void_t<decltype(interface::SingleConstant<std::decay_t<T>>::value)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a constant matrix, with the constant known at compile time.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept constant_matrix = requires { interface::SingleConstant<std::decay_t<T>>::value; };
#else
  constexpr bool constant_matrix = detail::is_constant_matrix<T>::value;
#endif


  /**
   * \brief The constant associated with T if T is a \ref constant_matrix.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct constant_coefficient;


#ifdef __cpp_concepts
  template<constant_matrix T>
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<constant_matrix<T>>>
#endif
  {
    // \todo Add some logic about, e.g., 1-by-1 constant diagonal matrices.
#  if __cpp_nontype_template_args >= 201911L
    static constexpr scalar_type_of_t<T> value = interface::SingleConstant<std::decay_t<T>>::value;
#  else
    static constexpr auto value = interface::SingleConstant<std::decay_t<T>>::value;
#  endif
  };


  /**
   * \brief Helper template for constant_coefficient.
   */
#ifdef __cpp_concepts
  template<constant_matrix T>
#else
  template<typename T>
#endif
  constexpr auto constant_coefficient_v = constant_coefficient<T>::value;


  // --------------------------------------------------------- //
  //  constant_diagonal_coefficient, constant_diagonal_matrix  //
  // --------------------------------------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_constant_diagonal_matrix : std::false_type {};

    template<typename T>
    struct is_constant_diagonal_matrix<T, std::void_t<
      decltype(interface::SingleConstantDiagonal<std::decay_t<T>>::value)>> : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a constant matrix, with the constant known at compile time.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept constant_diagonal_matrix = requires { interface::SingleConstantDiagonal<std::decay_t<T>>::value; };
#else
  constexpr bool constant_diagonal_matrix = detail::is_constant_diagonal_matrix<T>::value;
#endif


  /**
   * \brief The constant associated with a \ref constant_diagonal_matrix.
   * \details The constant must derive from std::integral_constant<type, value>
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct constant_diagonal_coefficient;


#ifdef __cpp_concepts
  template<constant_diagonal_matrix T>
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<constant_diagonal_matrix<T>>>
#endif
  {
    // \todo Add some logic about, e.g., 1-by-1 constant matrices.
#  if __cpp_nontype_template_args >= 201911L
    static constexpr scalar_type_of<T> value = interface::SingleConstantDiagonal<std::decay_t<T>>::value;
#  else
    static constexpr auto value = interface::SingleConstantDiagonal<std::decay_t<T>>::value;
#  endif
  };


  /// Helper template for constant_diagonal_coefficient.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix T>
#else
  template<typename T>
#endif
  constexpr auto constant_diagonal_coefficient_v = constant_diagonal_coefficient<T>::value;


  // ------------- //
  //  zero_matrix  //
  // ------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_zero_matrix : std::false_type {};

    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<
      are_within_tolerance(interface::SingleConstant<std::decay_t<T>>::value, 0)>>
      : std::true_type {};

    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<
      not constant_matrix<T> and are_within_tolerance(interface::SingleConstantDiagonal<std::decay_t<T>>::value, 0)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is known at compile time to be a constant matrix of value zero.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept zero_matrix = are_within_tolerance(constant_coefficient_v<T>, 0) or
    are_within_tolerance(constant_diagonal_coefficient_v<T>, 0);
#else
  constexpr bool zero_matrix = detail::is_zero_matrix<T>::value;
#endif


  // ----------------- //
  //  identity_matrix  //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_identity_matrix : std::false_type {};

    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<are_within_tolerance(constant_diagonal_coefficient<T>::value, 1)>>
      : std::true_type {};

    template<typename T, typename = void>
    struct is_1by1_identity_matrix : std::false_type {};

    template<typename T>
    struct is_1by1_identity_matrix<T, std::enable_if_t<are_within_tolerance(constant_coefficient<T>::value, 1) and
      row_dimension_of<T>::value == 1 and column_dimension_of<T>::value == 1>>
      : std::true_type {};
  }
#endif

  /**
   * \brief Specifies that a type is an identity matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept identity_matrix = are_within_tolerance(constant_diagonal_coefficient_v<T>, 1) or
    (are_within_tolerance(constant_coefficient_v<T>, 1) and row_dimension_of_v<T> == 1 and column_dimension_of_v<T> == 1);
#else
  constexpr bool identity_matrix = detail::is_identity_matrix<std::decay_t<T>>::value or
    detail::is_1by1_identity_matrix<std::decay_t<T>>::value;
#endif


  // ----------------- //
  //  diagonal_matrix  //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_inferred_diagonal_matrix : std::false_type {};

    template<typename T>
    struct is_inferred_diagonal_matrix<T, std::enable_if_t<
      row_dimension_of<T>::value == 1 and column_dimension_of<T>::value == 1>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a diagonal matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept diagonal_matrix = interface::DiagonalTraits<std::decay_t<T>>::is_diagonal or
    (interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::diagonal) or
    (interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
      interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::diagonal) or
    (not any_dynamic_dimension<T> and row_dimension_of_v<T> == column_dimension_of_v<T> and row_dimension_of_v<T> == 1) or
    constant_diagonal_matrix<T> or
    (zero_matrix<T> and not dynamic_rows<T> and row_dimension_of_v<T> == column_dimension_of_v<T>);
#else
  constexpr bool diagonal_matrix = interface::DiagonalTraits<std::decay_t<T>>::is_diagonal or
    (interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::diagonal) or
    (interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
      interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::diagonal) or
    detail::is_inferred_diagonal_matrix<T>::value or constant_diagonal_matrix<T> or
    (zero_matrix<T> and not dynamic_rows<T> and row_dimension_of_v<T> == column_dimension_of_v<T>);
#endif


  // ------------------------- //
  //  lower_triangular_matrix  //
  // ------------------------- //

  /**
   * \brief Specifies that a type is a lower-triangular matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept lower_triangular_matrix =
#else
  constexpr bool lower_triangular_matrix =
#endif
    (interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::lower) or diagonal_matrix<T>;


  // ------------------------- //
  //  upper_triangular_matrix  //
  // ------------------------- //

  /**
   * \brief Specifies that a type is an upper-triangular matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept upper_triangular_matrix =
#else
  constexpr bool upper_triangular_matrix =
#endif
    (interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::upper) or diagonal_matrix<T>;


  // ------------------- //
  //  triangular_matrix  //
  // ------------------- //

  /**
   * \brief Specifies that a type is a triangular matrix (upper or lower).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept triangular_matrix =
#else
  constexpr bool triangular_matrix =
#endif
    (interface::TriangularTraits<std::decay_t<T>>::triangle_type != TriangleType::none) or diagonal_matrix<T>;


  // ------------------ //
  //  triangle_type_of  //
  // ------------------ //

  /**
   * \brief The TriangleType associated with a \ref triangular_matrix.
   */
  template<typename T, typename...Ts>
  struct triangle_type_of
    : std::integral_constant<TriangleType,
      (diagonal_matrix<T> and ... and diagonal_matrix<Ts>) ? TriangleType::diagonal :
      ((lower_triangular_matrix<T> and ... and lower_triangular_matrix<Ts>) ? TriangleType::lower :
      ((upper_triangular_matrix<T> and ... and upper_triangular_matrix<Ts>) ? TriangleType::upper :
      TriangleType::none))> {};


  /**
   * \brief The TriangleType associated with a \ref triangular_matrix.
   */
  template<typename T, typename...Ts>
  constexpr auto triangle_type_of_v = triangle_type_of<T, Ts...>::value;


  // --------------------------- //
  //  lower_self_adjoint_matrix  //
  // --------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct imag_part_is_zero : std::false_type {};

    template<typename T>
    struct imag_part_is_zero<T, std::enable_if_t<std::imag(constant_coefficient<T>::value) == 0>>
      : std::true_type {};

    template<typename T, typename = void>
    struct diag_imag_part_is_zero : std::bool_constant<imag_part_is_zero<T>::value> {};

    template<typename T>
    struct diag_imag_part_is_zero<T, std::enable_if_t<std::imag(constant_diagonal_coefficient<T>::value) == 0>>
      : std::true_type {};

    template<typename T, typename = void>
    struct is_inferred_hermitian_matrix : std::false_type {};

    template<typename T>
    struct is_inferred_hermitian_matrix<T, std::enable_if_t<
      (not complex_number<typename scalar_type_of<T>::type> or zero_matrix<T> or diag_imag_part_is_zero<T>::value) and
      (diagonal_matrix<T> or
        (constant_matrix<T> and not any_dynamic_dimension<T> and row_dimension_of<T>::value == column_dimension_of<T>::value))>>
      : std::true_type {};
  };
#endif


  /**
   * \brief Specifies that T is an \ref eigen_self_adjoint_expr that stores data in the lower-left triangle.
   * \details This includes matrices that store data only along the diagonal, and is the default.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept lower_self_adjoint_matrix =
    (interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
      (interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::lower or
      interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::diagonal)) or
    ((not complex_number<scalar_type_of_t<T>> or zero_matrix<T> or
        std::imag(constant_coefficient_v<T>) == 0 or std::imag(constant_diagonal_coefficient_v<T>) == 0) and
      (diagonal_matrix<T> or (constant_matrix<T> and not any_dynamic_dimension<T> and row_dimension_of_v<T> == column_dimension_of_v<T>)));
#else
  constexpr bool lower_self_adjoint_matrix = (interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
      (interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::lower or
      interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::diagonal)) or
    detail::is_inferred_hermitian_matrix<T>::value;
#endif


  // --------------------------- //
  //  upper_self_adjoint_matrix  //
  // --------------------------- //

  /**
   * \brief Specifies that T is an \ref eigen_self_adjoint_expr that stores data in the upper-right triangle.
   * \details This includes matrices that store data only along the diagonal.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept upper_self_adjoint_matrix =
    (interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
      (interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::upper or
      interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::diagonal)) or
    ((not complex_number<scalar_type_of_t<T>> or zero_matrix<T> or
        std::imag(constant_coefficient_v<T>) == 0 or std::imag(constant_diagonal_coefficient_v<T>) == 0) and
      (diagonal_matrix<T> or (constant_matrix<T> and not any_dynamic_dimension<T> and row_dimension_of_v<T> == column_dimension_of_v<T>)));
#else
  constexpr bool upper_self_adjoint_matrix = (interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
      (interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::upper or
      interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::diagonal)) or
    detail::is_inferred_hermitian_matrix<T>::value;
#endif


  // --------------------- //
  //  self_adjoint_matrix  //
  // --------------------- //

  /**
   * \brief Specifies that a type is a self-adjoint matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept self_adjoint_matrix = interface::HermitianTraits<std::decay_t<T>>::is_hermitian or
    ((not complex_number<scalar_type_of_t<T>> or zero_matrix<T> or
        std::imag(constant_coefficient_v<T>) == 0 or std::imag(constant_diagonal_coefficient_v<T>) == 0) and
    (diagonal_matrix<T> or (constant_matrix<T> and not any_dynamic_dimension<T> and row_dimension_of_v<T> == column_dimension_of_v<T>)));
#else
  constexpr bool self_adjoint_matrix = interface::HermitianTraits<std::decay_t<T>>::is_hermitian or
    detail::is_inferred_hermitian_matrix<T>::value;
#endif


  // ------------------------------- //
  //  self_adjoint_triangle_type_of  //
  // ------------------------------- //

  /**
   * \brief The TriangleType associated with the storage triangle of one or more matrices.
   */
  template<typename T, typename...Ts>
  struct self_adjoint_triangle_type_of
    : std::integral_constant<TriangleType,
      (diagonal_matrix<T> and ... and diagonal_matrix<Ts>) ? TriangleType::diagonal :
      ((lower_self_adjoint_matrix<T> and ... and lower_self_adjoint_matrix<Ts>) ? TriangleType::lower :
      ((upper_self_adjoint_matrix<T> and ... and upper_self_adjoint_matrix<Ts>) ? TriangleType::upper :
      TriangleType::none))> {};


  /**
   * \brief The TriangleType associated with the storage triangle of a \ref self_adjoint_matrix.
   */
  template<typename T, typename...Ts>
  constexpr auto self_adjoint_triangle_type_of_v = self_adjoint_triangle_type_of<T, Ts...>::value;


  // --------------- //
  //  square_matrix  //
  // --------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_square_dimensions : std::false_type {};

    template<typename T>
    struct has_square_dimensions<T, std::enable_if_t<
      not any_dynamic_dimension<T> and row_dimension_of<T>::value == column_dimension_of<T>::value>>
      : std::true_type {};


    template<typename T, typename = void>
    struct has_equivalent_coefficients : std::true_type {};

    template<typename T>
    struct has_equivalent_coefficients<T, std::enable_if_t<
      not equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : std::false_type {};
  }
#endif


  /**
   * \brief Specifies that a matrix is square (i.e., has the same number and type of rows and column).
   * \details If T is a \ref typed_matrix, the row coefficients must also be \ref equivalent_to the column coefficients.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept square_matrix =
    (not any_dynamic_dimension<T> and (row_dimension_of_v<T> == column_dimension_of_v<T>) and
      (not requires { typename MatrixTraits<T>::RowCoefficients; typename MatrixTraits<T>::ColumnCoefficients; } or
        equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>)) or
    (any_dynamic_dimension<T> and (self_adjoint_matrix<T> or triangular_matrix<T>));
#else
  constexpr bool square_matrix =
    (detail::has_square_dimensions<std::decay_t<T>>::value and detail::has_equivalent_coefficients<T>::value) or
    (any_dynamic_dimension<T> and (self_adjoint_matrix<T> or triangular_matrix<T>));
#endif


  // ------------------- //
  //  one_by_one_matrix  //
  // ------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_one_by_one_matrix : std::false_type {};

    template<typename T>
    struct is_one_by_one_matrix<T, std::enable_if_t<
      (row_dimension_of<T>::value == 1 or column_dimension_of<T>::value == 1) and square_matrix<T>>> : std::true_type {};
  }
#endif

  /**
   * \brief Specifies that a type is a one-by-one matrix (i.e., one row and one column).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept one_by_one_matrix = (row_dimension_of_v<T> == 1 or column_dimension_of_v<T> == 1) and square_matrix<T>;
#else
  constexpr bool one_by_one_matrix = detail::is_one_by_one_matrix<T>::value;
#endif


  // ---------------------- //
  //   mean, wrapped_mean   //
  // ---------------------- //

  namespace internal
  {
    template<typename T>
    struct is_mean : std::false_type {};
  }


  /**
   * \brief Specifies that T is a mean (i.e., is a specialization of the class Mean).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept mean = internal::is_mean<std::decay_t<T>>::value;
#else
  constexpr bool mean = internal::is_mean<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_wrapped_mean : std::false_type {}; //< see forward-class-declarations.hpp


    template<typename T>
    struct is_wrapped_mean<T, std::enable_if_t<mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T is a wrapped mean (i.e., its row coefficients have at least one type that requires wrapping).
   */
#ifdef __cpp_concepts
  template<typename T>
  concept wrapped_mean = mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only);
#else
  template<typename T>
  constexpr bool wrapped_mean = detail::is_wrapped_mean<T>::value;
#endif


  // ----------------------------------------- //
  //   euclidean_mean, euclidean_transformed   //
  // ----------------------------------------- //

  namespace internal
  {
    template<typename T>
    struct is_euclidean_mean : std::false_type {};
  }


  /**
   * \brief Specifies that T is a Euclidean mean (i.e., is a specialization of the class EuclideanMean).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_mean = internal::is_euclidean_mean<std::decay_t<T>>::value;
#else
  constexpr bool euclidean_mean = internal::is_euclidean_mean<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_euclidean_transformed : std::false_type {};


    template<typename T>
    struct is_euclidean_transformed<T, std::enable_if_t<
      euclidean_mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T is a Euclidean mean that actually has coefficients that are transformed to Euclidean space.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_transformed = euclidean_mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only);
#else
  template<typename T>
  constexpr bool euclidean_transformed = detail::is_euclidean_transformed<T>::value;
#endif


  // ---------------- //
  //   typed matrix   //
  // ---------------- //

  namespace internal
  {
    template<typename T>
    struct is_matrix : std::false_type {};
  }


  /**
   * \brief Specifies that T is a typed matrix (i.e., is a specialization of Matrix, Mean, or EuclideanMean).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix = mean<T> or euclidean_mean<T> or internal::is_matrix<std::decay_t<T>>::value;
#else
  constexpr bool typed_matrix = mean<T> or euclidean_mean<T> or internal::is_matrix<std::decay_t<T>>::value;
#endif


  // ------------------------------------------------------------ //
  //   untyped_columns, column_vector, untyped_rows, row_vector   //
  // ------------------------------------------------------------ //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_untyped_columns : std::true_type {};

    template<typename T>
    struct has_untyped_columns<T, std::void_t<typename MatrixTraits<T>::ColumnCoefficients>>
      : std::bool_constant<MatrixTraits<T>::ColumnCoefficients::axes_only> {};
  }
#endif


  /**
   * \brief Specifies that T has untyped (or Axis typed) column coefficients.
   * \details T must be either a native matrix or its columns must all have type Axis.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept untyped_columns = (not requires { typename MatrixTraits<T>::ColumnCoefficients; }) or
    (MatrixTraits<T>::ColumnCoefficients::axes_only);
#else
  template<typename T>
  constexpr bool untyped_columns = detail::has_untyped_columns<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_column_vector : std::false_type {};

    template<typename T>
    struct is_column_vector<T, std::enable_if_t<untyped_columns<T> and column_dimension_of<T>::value == 1>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T is a column vector (i.e., has one untyped or Axis-typed column).
   * \details If T is a typed_matrix, its column must be of type Axis.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept column_vector = untyped_columns<T> and (column_dimension_of_v<T> == 1);
#else
  template<typename T>
  constexpr bool column_vector = detail::is_column_vector<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_untyped_rows : std::true_type {};

    template<typename T>
    struct has_untyped_rows<T, std::void_t<typename MatrixTraits<T>::RowCoefficients>>
      : std::bool_constant<MatrixTraits<T>::RowCoefficients::axes_only> {};
  }
#endif


  /**
   * \brief Specifies that T has untyped (or Axis typed) row bases.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept untyped_rows = (not requires { typename MatrixTraits<T>::RowCoefficients; }) or
    (MatrixTraits<T>::RowCoefficients::axes_only);
#else
  template<typename T>
  constexpr bool untyped_rows = detail::has_untyped_rows<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_row_vector : std::false_type {};

    template<typename T>
    struct is_row_vector<T, std::enable_if_t<untyped_rows<T> and row_dimension_of<T>::value == 1>> : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T is a row vector (i.e., has one untyped or Axis-typed row).
   * \details If T is a typed_matrix, its row must be of type Axis.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept row_vector = untyped_rows<T> and (row_dimension_of_v<T> == 1);
#else
  template<typename T>
  constexpr bool row_vector = detail::is_row_vector<std::decay_t<T>>::value;
#endif


  // ------------------------------------------------------------- //
  //  covariance, self_adjoint_covariance, triangular_covariance  //
  // ------------------------------------------------------------- //

  namespace internal
  {
    template<typename T>
    struct is_self_adjoint_covariance : std::false_type {};
  }


  /**
   * \brief T is a self-adjoint covariance matrix (i.e., a specialization of Covariance).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept self_adjoint_covariance = internal::is_self_adjoint_covariance<std::decay_t<T>>::value;
#else
  constexpr bool self_adjoint_covariance = internal::is_self_adjoint_covariance<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    template<typename T>
    struct is_triangular_covariance : std::false_type {};
  }


  /**
   * \brief T is a square root (Cholesky) covariance matrix (i.e., a specialization of SquareRootCovariance).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept triangular_covariance = internal::is_triangular_covariance<std::decay_t<T>>::value;
#else
  constexpr bool triangular_covariance = internal::is_triangular_covariance<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is a specialization of either Covariance or SquareRootCovariance.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance = self_adjoint_covariance<T> or triangular_covariance<T>;
#else
  constexpr bool covariance = self_adjoint_covariance<T> or triangular_covariance<T>;
#endif


  // --------------- //
  //  distributions  //
  // --------------- //

  namespace internal
  {
    template<typename T>
    struct is_gaussian_distribution : std::false_type {};
  }

  /**
   * \brief T is a Gaussian distribution.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept gaussian_distribution = internal::is_gaussian_distribution<std::decay_t<T>>::value;
#else
  constexpr bool gaussian_distribution = internal::is_gaussian_distribution<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is a statistical distribution of any kind that is defined in OpenKalman.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept distribution = gaussian_distribution<T>;
#else
  constexpr bool distribution = gaussian_distribution<T>;
#endif


  // --------------- //
  //  cholesky_form  //
  // --------------- //

  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_cholesky_form : std::false_type {};

    template<typename T>
    struct is_cholesky_form<T, std::enable_if_t<covariance<T>>>
      : std::bool_constant<not self_adjoint_matrix<nested_matrix_of_t<T>>> {};

    template<typename T>
    struct is_cholesky_form<T, std::enable_if_t<distribution<T>>>
      : is_cholesky_form<typename DistributionTraits<T>::Covariance> {};
#endif
  }


  /**
   * \brief Specifies that a type has a nested native matrix that is a Cholesky square root.
   * \details If this is true, then nested_matrix_of_t<T> is true.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept cholesky_form = (not covariance<T> or not self_adjoint_matrix<nested_matrix_of_t<T>>) or
    (not distribution<T> or not self_adjoint_matrix<nested_matrix_of_t<typename DistributionTraits<T>::Covariance>>);
#else
  constexpr bool cholesky_form = detail::is_cholesky_form<std::decay_t<T>>::value;
#endif


  // ------------------------- //
  //    covariance_nestable    //
  // ------------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether T can be wrapped in a covariance.
     * \note: This class should be specialized for all appropriate matrix classes.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_covariance_nestable : std::false_type {};
  }

  /**
   * \brief T is an acceptable nested matrix for a covariance (including triangular_covariance).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance_nestable = internal::is_covariance_nestable<std::decay_t<T>>::value;
#else
  constexpr bool covariance_nestable = internal::is_covariance_nestable<std::decay_t<T>>::value;
#endif


  // --------------------------- //
  //    typed_matrix_nestable    //
  // --------------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether T is acceptable to be nested in a typed_matrix.
     * \note: This class should be specialized for all appropriate matrix classes.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_typed_matrix_nestable : std::false_type {};
  }

  /**
   * \brief Specifies a type that is nestable in a general typed matrix (e.g., matrix, mean, or euclidean_mean)
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix_nestable = internal::is_typed_matrix_nestable<std::decay_t<T>>::value;
#else
  constexpr bool typed_matrix_nestable = internal::is_typed_matrix_nestable<std::decay_t<T>>::value;
#endif


  // ---------- //
  //  writable  //
  // ---------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_writable : std::false_type {};
  }

  /**
   * \internal
   * \brief Specifies that T is a writable matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept writable =
#else
  constexpr bool writable =
#endif
    internal::is_writable<std::decay_t<T>>::value and (not std::is_const_v<std::remove_reference_t<T>>);


  // ------------ //
  //  modifiable  //
  // ------------ //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_modifiable : std::true_type {};


    // Custom modifiability parameter that can be defined in the native matrix ecosystem.
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_modifiable_native : std::true_type {};

  } // namespace internal


  /**
   * \internal
   * \brief Specifies that U is not obviously incompatible with T, such that assigning U to T might be possible.
   * \details The result is true unless there is an incompatibility of some kind that would prevent assignment.
   * Examples of such incompatibility are if T is constant or has a nested constant type, if T and U have a
   * different shape or scalar type, or if T and U differ as to being self-adjoint, triangular, diagonal,
   * zero, or identity. Even if this concept is true, a compile-time error is still possible.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept modifiable =
#else
  constexpr bool modifiable =
#endif
    internal::is_modifiable<T, U>::value and internal::is_modifiable_native<T, U>::value;

} // namespace OpenKalman

#endif //OPENKALMAN_FORWARD_TRAITS_HPP
