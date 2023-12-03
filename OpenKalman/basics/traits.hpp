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
 * \brief Declarations for OpenKalman and native-matrix traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  // --------------------------------------- //
  //  constant-related classes and concepts  //
  // --------------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
    concept has_get_constant_interface = requires(T t) {
      {interface::indexible_object_traits<std::decay_t<T>>::get_constant(t)} -> scalar_constant<c>;
    };
#else
    template<typename T, typename = void>
    struct get_constant_res {};

    template<typename T>
    struct get_constant_res<T, std::void_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant(std::declval<T>()))>>
    {
      using type = std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant(std::declval<T>()))>;
    };

    template<typename T, CompileTimeStatus c, typename = void>
    struct has_get_constant_interface_impl : std::false_type {};

    template<typename T, CompileTimeStatus c>
    struct has_get_constant_interface_impl<T, c, std::enable_if_t<scalar_constant<typename get_constant_res<T>::type, c>>>
      : std::true_type {};

    template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
    constexpr bool has_get_constant_interface = has_get_constant_interface_impl<T, c>::value;
#endif


#ifdef __cpp_concepts
    template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
    concept has_get_constant_diagonal_interface = requires(T t) {
      {interface::indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(t)} -> scalar_constant<c>;
    };
#else
    template<typename T, typename = void>
    struct get_constant_diagonal_res {};

    template<typename T>
    struct get_constant_diagonal_res<T, std::void_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(std::declval<T>()))>>
    {
      using type = std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(std::declval<T>()))>;
    };

    template<typename T, CompileTimeStatus c, typename = void>
    struct has_get_constant_diagonal_interface_impl : std::false_type {};

    template<typename T, CompileTimeStatus c>
    struct has_get_constant_diagonal_interface_impl<T, c, std::void_t<typename get_constant_diagonal_res<T>::type>>
      : std::bool_constant<scalar_constant<typename get_constant_diagonal_res<T>::type, c>> {};

    template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
    constexpr bool has_get_constant_diagonal_interface = has_get_constant_diagonal_interface_impl<std::decay_t<T>, c>::value;

    template<typename T, typename = void>
    struct constant_diagonal_matrix_is_zero : std::false_type {};

    template<typename T>
    struct constant_diagonal_matrix_is_zero<T, std::void_t<decltype(get_constant_diagonal_res<T>::type::value)>>
      : std::bool_constant<are_within_tolerance(get_constant_diagonal_res<T>::type::value, 0)> {};
#endif


#ifdef __cpp_concepts
    template<typename T>
    concept known_constant = detail::has_get_constant_interface<T, CompileTimeStatus::known> or
      (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::known> and
        (one_by_one_matrix<T, Likelihood::maybe> or requires(T t) {
          requires are_within_tolerance(std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(t))>::value, 0);
        }));
#else
    template<typename T, typename = void>
    struct is_known_constant : std::false_type {};

    template<typename T>
    struct is_known_constant<T, std::enable_if_t<detail::has_get_constant_interface<T, CompileTimeStatus::known> or
      (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::known> and
        (one_by_one_matrix<T, Likelihood::maybe> or detail::constant_diagonal_matrix_is_zero<T>::value))>>
      : std::true_type {};

    template<typename T>
    constexpr bool known_constant = is_known_constant<T>::value;
#endif


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct constant_status : std::integral_constant<Likelihood, Likelihood::definitely> {};

#ifdef __cpp_concepts
    template<typename T> requires (std::decay_t<T>::status == Likelihood::maybe)
    struct constant_status<T>
#else
    template<typename T>
    struct constant_status<T, std::enable_if_t<std::decay_t<T>::status == Likelihood::maybe>>
#endif
      : std::integral_constant<Likelihood, Likelihood::maybe> {};

  } // namespace detail


  // -------------------------------------- //
  //  constant_coefficient specializations  //
  // -------------------------------------- //

  /**
   * \internal
   * \brief Specialization of \ref constant_coefficient in which the constant is known at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::known_constant<T>
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<indexible<T> and detail::known_constant<T>>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

  public:

    constexpr constant_coefficient() = default;

    explicit constexpr constant_coefficient(const std::decay_t<T>&) {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_coefficient;

    static constexpr value_type value = []{
      if constexpr (detail::has_get_constant_interface<T, CompileTimeStatus::known>)
        return std::decay_t<decltype(Trait::get_constant(std::declval<T>()))>::value;
      else
        return std::decay_t<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
    }();

    static constexpr Likelihood status = []{
      if constexpr (not has_dynamic_dimensions<T>)
        return Likelihood::definitely;
      else if constexpr (detail::has_get_constant_interface<T, CompileTimeStatus::known>)
        return detail::constant_status<decltype(Trait::get_constant(std::declval<T>()))>::value;
      else
        return Likelihood::maybe;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }

    constexpr auto operator+() { return *this; }

    constexpr auto operator-() { return internal::scalar_constant_operation {std::negate<>{}, *this}; }


#ifdef __cpp_concepts
    constexpr auto operator+(const scalar_constant<CompileTimeStatus::known> auto& arg)
#else
    template<typename Arg, std::enable_if_t<scalar_constant<Arg, CompileTimeStatus::known>, int> = 0>
    constexpr auto operator+(const Arg& arg)
#endif
    {
      return internal::scalar_constant_operation {std::plus<>{}, *this, arg};
    }


#ifdef __cpp_concepts
    constexpr auto operator-(const scalar_constant<CompileTimeStatus::known> auto& arg)
#else
    template<typename Arg, std::enable_if_t<scalar_constant<Arg, CompileTimeStatus::known>, int> = 0>
    constexpr auto operator-(const Arg& arg)
#endif
    {
      return internal::scalar_constant_operation {std::minus<>{}, *this, arg};
    }


#ifdef __cpp_concepts
    constexpr auto operator*(const scalar_constant<CompileTimeStatus::known> auto& arg)
#else
    template<typename Arg, std::enable_if_t<scalar_constant<Arg, CompileTimeStatus::known>, int> = 0>
    constexpr auto operator*(const Arg& arg)
#endif
    {
      return internal::scalar_constant_operation {std::multiplies<>{}, *this, arg};
    }


#ifdef __cpp_concepts
    template<scalar_constant<CompileTimeStatus::known> Arg>
#else
    template<typename Arg, std::enable_if_t<scalar_constant<Arg, CompileTimeStatus::known>, int> = 0>
#endif
    constexpr auto operator/(const Arg& arg)
    {
      return internal::scalar_constant_operation {std::divides<>{}, *this, arg};
    }

  };


  /**
   * \internal
   * \brief Specialization of \ref constant_coefficient in which the constant is unknown at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not detail::known_constant<T>) and
    (detail::has_get_constant_interface<T, CompileTimeStatus::unknown> or one_by_one_matrix<T, Likelihood::maybe>)
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<(not detail::known_constant<T>) and
    (detail::has_get_constant_interface<T, CompileTimeStatus::unknown> or one_by_one_matrix<T, Likelihood::maybe>)>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

    template<typename Arg, std::size_t...Ix>
    static constexpr auto get_zero_component(const Arg& arg, std::index_sequence<Ix...>) { return get_element(arg, static_cast<decltype(Ix)>(0)...); }

  public:

    explicit constexpr constant_coefficient(const std::decay_t<T>& t) : value {[](const auto& t){
        if constexpr (detail::has_get_constant_interface<T, CompileTimeStatus::unknown>)
          return get_scalar_constant_value(Trait::get_constant(t));
        else if constexpr (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::unknown>)
          return get_scalar_constant_value(Trait::get_constant_diagonal(t));
        else
          return get_zero_component(t, std::make_index_sequence<index_count_v<T>>{});
      }(t)} {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_coefficient;

    static constexpr Likelihood status = []{
      if constexpr (not has_dynamic_dimensions<T>)
        return Likelihood::definitely;
      else if constexpr (detail::has_get_constant_interface<T, CompileTimeStatus::unknown>)
        return detail::constant_status<decltype(Trait::get_constant(std::declval<T>()))>::value;
      else
        return Likelihood::maybe;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }

  private:
    value_type value;
  };


  // ----------------------------------------------- //
  //  constant_diagonal_coefficient specializations  //
  // ----------------------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
    concept known_constant_diagonal = detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::known> or
      (detail::has_get_constant_interface<T, CompileTimeStatus::known> and
        (one_by_one_matrix<T, Likelihood::maybe> or (square_matrix<T, Likelihood::maybe> and
          requires(T t) {
            requires are_within_tolerance(std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant(t))>::value, 0);
          })));
#else
    template<typename T, typename = void>
    struct constant_matrix_is_zero : std::false_type {};

    template<typename T>
    struct constant_matrix_is_zero<T, std::enable_if_t<are_within_tolerance(get_constant_res<T>::type::value, 0)>>
      : std::true_type {};


    template<typename T, typename = void>
    struct is_known_constant_diagonal : std::false_type {};

    template<typename T>
    struct is_known_constant_diagonal<T, std::enable_if_t<
      detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::known> or
      (detail::has_get_constant_interface<T, CompileTimeStatus::known> and
        (one_by_one_matrix<T, Likelihood::maybe> or (square_matrix<T, Likelihood::maybe> and
        detail::constant_matrix_is_zero<T>::value)))>>
      : std::true_type {};

    template<typename T>
    constexpr bool known_constant_diagonal = is_known_constant_diagonal<T>::value;
#endif
  } // namepsace detail


  /**
   * \internal
   * \brief Specialization of \ref constant_diagonal_coefficient in which the constant is known at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::known_constant_diagonal<T>
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<indexible<T> and detail::known_constant_diagonal<T>>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

  public:

    constexpr constant_diagonal_coefficient() = default;

    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>&) {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_diagonal_coefficient;

    static constexpr value_type value = []{
      if constexpr (detail::has_get_constant_diagonal_interface<T>)
        return std::decay_t<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
      else
        return std::decay_t<decltype(Trait::get_constant(std::declval<T>()))>::value;
    }();

    static constexpr Likelihood status = []{
      if constexpr (not has_dynamic_dimensions<T>)
        return Likelihood::definitely;
      else if constexpr (detail::has_get_constant_diagonal_interface<T>)
        return detail::constant_status<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
      else
        return Likelihood::maybe;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }

    constexpr auto operator+() { return *this; }

    constexpr auto operator-()
    {
      if constexpr (value == 0) return *this;
      else return internal::scalar_constant_operation {std::negate<>{}, *this};
    }


#ifdef __cpp_concepts
    constexpr auto operator+(const scalar_constant<CompileTimeStatus::known> auto& arg)
#else
    template<typename Arg, std::enable_if_t<scalar_constant<Arg, CompileTimeStatus::known>, int> = 0>
    constexpr auto operator+(const Arg& arg)
#endif
    {
      return internal::scalar_constant_operation {std::plus<>{}, *this, arg};
    }


#ifdef __cpp_concepts
    constexpr auto operator-(const scalar_constant<CompileTimeStatus::known> auto& arg)
#else
    template<typename Arg, std::enable_if_t<scalar_constant<Arg, CompileTimeStatus::known>, int> = 0>
    constexpr auto operator-(const Arg& arg)
#endif
    {
      return internal::scalar_constant_operation {std::minus<>{}, *this, arg};
    }


#ifdef __cpp_concepts
    constexpr auto operator*(const scalar_constant<CompileTimeStatus::known> auto& arg)
#else
    template<typename Arg, std::enable_if_t<scalar_constant<Arg, CompileTimeStatus::known>, int> = 0>
    constexpr auto operator*(const Arg& arg)
#endif
    {
      return internal::scalar_constant_operation {std::multiplies<>{}, *this, arg};
    }


#ifdef __cpp_concepts
    template<scalar_constant<CompileTimeStatus::known> Arg>
#else
    template<typename Arg, std::enable_if_t<scalar_constant<Arg, CompileTimeStatus::known>, int> = 0>
#endif
    constexpr auto operator/(const Arg& arg)
    {
      return internal::scalar_constant_operation {std::divides<>{}, *this, arg};
    }

  };


  /**
   * \internal
   * \brief Specialization of \ref constant_diagonal_coefficient in which the constant can be known only at runtime.
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not detail::known_constant_diagonal<T>) and
    (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::unknown> or one_by_one_matrix<T, Likelihood::maybe>)
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<indexible<T> and (not detail::known_constant_diagonal<T>) and
    (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::unknown> or one_by_one_matrix<T, Likelihood::maybe>)>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

    template<typename Arg, std::size_t...Ix>
    static constexpr auto get_zero_component(const Arg& arg, std::index_sequence<Ix...>) { return get_element(arg, static_cast<decltype(Ix)>(0)...); }

  public:

    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>& t) : value {[](const auto& t){
        if constexpr (detail::has_get_constant_diagonal_interface<T>)
          return get_scalar_constant_value(Trait::get_constant_diagonal(t));
        else if constexpr (detail::has_get_constant_interface<T>)
          return get_scalar_constant_value(Trait::get_constant(t));
        else
          return get_zero_component(t, std::make_index_sequence<index_count_v<T>>{});
      }(t)} {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_diagonal_coefficient;

    static constexpr Likelihood status = []{
      if constexpr (not has_dynamic_dimensions<T>)
        return Likelihood::definitely;
      else if constexpr (detail::has_get_constant_diagonal_interface<T>)
        return detail::constant_status<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
      else
        return Likelihood::maybe;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }

  private:
    value_type value;
  };


  // ----------------------------------------------------------------------- //
  //  arithmetic for constant_coefficient and constant_diagonal_coefficient  //
  // ----------------------------------------------------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_internal_constant : std::false_type {};

    template<typename T>
    struct is_internal_constant<constant_coefficient<T>> : std::true_type {};

    template<typename T>
    struct is_internal_constant<constant_diagonal_coefficient<T>> : std::true_type {};

    template<Likelihood b, typename C, auto...constant>
    struct is_internal_constant<internal::ScalarConstant<b, C, constant...>> : std::true_type {};

    template<typename Operation, typename...Ts>
    struct is_internal_constant<internal::scalar_constant_operation<Operation, Ts...>> : std::true_type {};


    template<typename T>
#ifdef __cpp_concepts
    concept internal_constant =
#else
    constexpr bool internal_constant =
#endif
      is_internal_constant<std::decay_t<T>>::value;

  } // namespace detail


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires detail::internal_constant<Arg1> or detail::internal_constant<Arg2>
  constexpr auto operator+(const Arg1& arg1, const Arg2& arg2)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::internal_constant<Arg1> or detail::internal_constant<Arg2>, int> = 0>
  constexpr auto operator+(const Arg1& arg1, const Arg2& arg2)
#endif
  {
    return internal::scalar_constant_operation {std::plus<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires detail::internal_constant<Arg1> or detail::internal_constant<Arg2>
  constexpr auto operator-(const Arg1& arg1, const Arg2& arg2)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::internal_constant<Arg1> or detail::internal_constant<Arg2>, int> = 0>
  constexpr auto operator-(const Arg1& arg1, const Arg2& arg2)
#endif
  {
    return internal::scalar_constant_operation {std::minus<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires detail::internal_constant<Arg1> or detail::internal_constant<Arg2>
  constexpr auto operator*(const Arg1& arg1, const Arg2& arg2)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::internal_constant<Arg1> or detail::internal_constant<Arg2>, int> = 0>
  constexpr auto operator*(const Arg1& arg1, const Arg2& arg2)
#endif
  {
    return internal::scalar_constant_operation {std::multiplies<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires detail::internal_constant<Arg1> or detail::internal_constant<Arg2>
  constexpr auto operator/(const Arg1& arg1, const Arg2& arg2)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::internal_constant<Arg1> or detail::internal_constant<Arg2>, int> = 0>
  constexpr auto operator/(const Arg1& arg1, const Arg2& arg2)
#endif
  {
    return internal::scalar_constant_operation {std::divides<>{}, arg1, arg2};
  }


  // --------------- //
  //  typed_adapter  //
  // --------------- //

  /**
   * \brief Specifies that T is a typed adapter expression.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_adapter =
#else
  constexpr bool typed_adapter =
#endif
    typed_matrix<T> or covariance<T> or euclidean_expr<T>;


  // ----------------- //
  //  untyped_adapter  //
  // ----------------- //

  /**
   * \brief Specifies that T is an untyped adapter expression.
   * \details Untyped adapter expressions are generally used whenever the native matrix library does not have an
   * important built-in matrix type, such as a diagonal matrix, a triangular matrix, or a hermitian matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept untyped_adapter =
#else
  constexpr bool untyped_adapter =
#endif
    eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>;


  // ========= //
  //  Aliases  //
  // ========= //

  // ------------------------- //
  //  dense_writable_matrix_t  //
  // ------------------------- //

  namespace detail
  {
    template<typename T, Layout layout, typename Scalar, typename...D>
    struct dense_writable_matrix_impl
    {
      using type = std::decay_t<decltype(make_default_dense_writable_matrix_like<T, layout, Scalar>(std::declval<D>()...))>;
    };


    template<typename T, Layout layout, typename Scalar>
    struct dense_writable_matrix_impl<T, layout, Scalar>
    {
      using type = std::decay_t<decltype(make_default_dense_writable_matrix_like<layout, Scalar>(std::declval<T>()))>;
    };
  }


  /**
   * \brief An alias for a dense, writable matrix, patterned on parameter T.
   * \tparam T A matrix or array from the relevant matrix library.
   * \tparam S A scalar type (may or may not be </code>scalar_type_of_t<T></code>.
   * \tparam layout The /ref Layout of the result.
   * \tparam D \ref vector_space_descriptor objects defining the dimensions of the new matrix.
   * \todo Create typed Matrix if Ds are typed.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type S = scalar_type_of_t<T>, vector_space_descriptor...D>
    requires (layout != Layout::stride)
#else
  template<typename T, Layout layout = Layout::none, typename S = scalar_type_of_t<T>, typename...D>
#endif
  using dense_writable_matrix_t = typename detail::dense_writable_matrix_impl<T, layout, std::decay_t<S>, D...>::type;


  // --------------------------------- //
  //  untyped_dense_writable_matrix_t  //
  // --------------------------------- //

  /**
   * \brief An alias for a dense, writable matrix, patterned on parameter T.
   * \tparam T A matrix or array from the relevant matrix library.
   * \tparam layout The /ref Layout of the result.
   * \tparam S A scalar type (may or may not be </code>scalar_type_of_t<T></code>.
   * \tparam D Integral values defining the dimensions of the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type S = scalar_type_of_t<T>, std::integral auto...D> requires
    ((std::is_integral_v<decltype(D)> and D >= 0) and ...) and (layout != Layout::stride)
#else
  template<typename T, Layout layout = Layout::none, typename S = scalar_type_of_t<T>, auto...D>
#endif
  using untyped_dense_writable_matrix_t = dense_writable_matrix_t<T, layout, S, Dimensions<static_cast<const std::size_t>(D)>...>;


  // --------------------------- //
  //  equivalent_self_contained  //
  // --------------------------- //

  /**
   * \brief An alias for type, derived from and equivalent to parameter T, that is self-contained.
   * \details Use this alias to obtain a type, equivalent to T, that can safely be returned from a function.
   * \sa self_contained, make_self_contained
   * \internal \sa interface::indexible_object_traits
   */
  template<typename T>
  using equivalent_self_contained_t = std::remove_reference_t<decltype(make_self_contained(std::declval<T>()))>;


  // ------------ //
  //  passable_t  //
  // ------------ //

  /**
   * \brief An alias for a type, derived from and equivalent to parameter T, that can be passed as a function parameter.
   * \tparam T The type in question.
   * \details A passable type T is either an lvalue reference or is \ref equivalent_self_contained_t.
   */
  template<typename T>
  using passable_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, equivalent_self_contained_t<T>>;


  namespace internal
  {
    // --------------- //
    //  is_modifiable  //
    // --------------- //

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct has_const : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires std::is_const_v<std::remove_reference_t<T>> or
      (requires { typename nested_matrix_of_t<T>; } and has_const<nested_matrix_of_t<T>>::value)
    struct has_const<T> : std::true_type {};
#else
    template<typename T>
    struct has_const<T, std::enable_if_t<std::is_const_v<std::remove_reference_t<T>>>> : std::true_type {};

    template<typename T>
    struct has_const<T, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>) and
      has_const<nested_matrix_of_t<T>>::value>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      has_const<T>::value or
      (not maybe_has_same_shape_as<T, U>) or
      (not std::same_as<scalar_type_of_t<T>, scalar_type_of_t<U>>) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (triangular_matrix<T, TriangleType::upper> and not triangular_matrix<U, TriangleType::upper>) or
      (triangular_matrix<T, TriangleType::lower> and not triangular_matrix<U, TriangleType::lower>) or
      (hermitian_matrix<T> and not hermitian_matrix<U>)
    struct is_modifiable<T, U> : std::false_type {};
#else
    template<typename T, typename U>
    struct is_modifiable<T, U, std::enable_if_t<
      has_const<T>::value or
      (not maybe_has_same_shape_as<T, U>) or
      (not std::is_same_v<scalar_type_of_t<T>, scalar_type_of_t<U>>) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (triangular_matrix<T, TriangleType::upper> and not triangular_matrix<U, TriangleType::upper>) or
      (triangular_matrix<T, TriangleType::lower> and not triangular_matrix<U, TriangleType::lower>) or
      (hermitian_matrix<T> and not hermitian_matrix<U>)>> : std::false_type {};
#endif

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_TRAITS_HPP
