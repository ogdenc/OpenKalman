/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded general linear-algebra functions.
 */

#ifndef OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP

#include<complex>


namespace OpenKalman
{
  /**
   * \brief Take the conjugate of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr decltype(auto) conjugate(Arg&& arg) noexcept
  {
    if constexpr (not complex_number<scalar_type_of_t<Arg>> or zero_matrix<Arg> or identity_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (real_axis_number<constant_coefficient<Arg>>)
        return std::forward<Arg>(arg);
      else
        return make_constant_matrix_like(internal::constexpr_conj(constant_coefficient{arg}), std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (real_axis_number<constant_diagonal_coefficient<Arg>>)
        return std::forward<Arg>(arg);
      else
        return to_diagonal(make_constant_matrix_like(internal::constexpr_conj(constant_diagonal_coefficient{arg}),
          diagonal_of(std::forward<Arg>(arg))));
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return to_diagonal(conjugate(diagonal_of(std::forward<Arg>(arg))));
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
  }


  namespace detail
  {
    template<typename C, typename Arg, std::size_t...Is>
    constexpr decltype(auto) transpose_constant(C&& c, Arg&& arg, std::index_sequence<Is...>) noexcept
    {
      return make_constant_matrix_like<Arg>(std::forward<C>(c),
        get_vector_space_descriptor<1>(arg), get_vector_space_descriptor<0>(arg), get_vector_space_descriptor<Is + 2>(arg)...);
    }
  }


  /**
   * \brief Take the transpose of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires (max_tensor_order_of_v<Arg> <= 2)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (max_tensor_order_of_v<Arg> <= 2), int> = 0>
#endif
  constexpr decltype(auto) transpose(Arg&& arg) noexcept
  {
    if constexpr (diagonal_matrix<Arg> or (hermitian_matrix<Arg> and not complex_number<scalar_type_of_t<Arg>>) or
      (constant_matrix<Arg> and square_matrix<Arg>))
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr std::make_index_sequence<std::max({index_count_v<Arg>, static_cast<std::size_t>(2)}) - 2> seq;
      return detail::transpose_constant(constant_coefficient{arg}, std::forward<Arg>(arg), seq);
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::transpose(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the adjoint of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires (max_tensor_order_of_v<Arg> <= 2)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (max_tensor_order_of_v<Arg> <= 2), int> = 0>
#endif
  constexpr decltype(auto) adjoint(Arg&& arg) noexcept
  {
    if constexpr (hermitian_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (real_axis_number<constant_coefficient<Arg>>)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not has_dynamic_dimensions<Arg> and index_dimension_of_v<Arg, 0> == index_dimension_of_v<Arg, 1>)
        return conjugate(std::forward<Arg>(arg));
      else
      {
        constexpr std::make_index_sequence<std::max({index_count_v<Arg>, static_cast<std::size_t>(2)}) - 2> seq;
        return detail::transpose_constant(internal::constexpr_conj(constant_coefficient{arg}), std::forward<Arg>(arg), seq);
      }
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the determinant of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> Arg> requires (max_tensor_order_of_v<Arg> <= 2)
  constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
  template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and (max_tensor_order_of_v<Arg> <= 2), int> = 0>
  constexpr auto
#endif
  determinant(Arg&& arg)
  {
    constexpr auto ix = []{ if constexpr (dynamic_dimension<Arg, 0>) return 1; else return 0; }();

    if constexpr (identity_matrix<Arg>)
    {
      return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 1>{};
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return internal::constexpr_pow(constant_diagonal_coefficient{arg}, internal::index_dimension_scalar_constant_of<ix>(arg))();
    }
    else if constexpr (dimension_size_of_index_is<Arg, 0, 1> or dimension_size_of_index_is<Arg, 1, 1>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'determinant' is not a square matrix"};
      return constant_coefficient {arg};
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'determinant' is not a square matrix"};
      return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0>{};
    }
    else if constexpr (dimension_size_of_index_is<Arg, 0, 0> or dimension_size_of_index_is<Arg, 1, 0>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'determinant' is not a square matrix"};
      return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 1>{};
    }
    else if constexpr (triangular_matrix<Arg> and not dynamic_dimension<Arg, ix> and index_dimension_of_v<Arg, ix> >= 2) // this includes the diagonal case
    {
      return reduce(std::multiplies<scalar_type_of_t<Arg>>{}, diagonal_of(std::forward<Arg>(arg)));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (has_dynamic_dimensions<Arg>)
      {
        auto d = get_is_square(arg);
        if (not d) throw std::invalid_argument{"Argument of 'determinant' is not a square matrix."};
        else if (*d >= 2) return static_cast<scalar_type_of_t<Arg>>(0);
        else if (*d == 1) return static_cast<scalar_type_of_t<Arg>>(constant_coefficient {arg});
        else return static_cast<scalar_type_of_t<Arg>>(1); // empty matrix
      }
      else
      {
        return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0>{};
      }
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::determinant(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  /**
   * \brief Take the trace of a matrix
   * \tparam Arg The matrix
   * \todo Redefine as a particular tensor contraction.
   */
  template<square_matrix<Likelihood::maybe> Arg> requires (max_tensor_order_of_v<Arg> <= 2)
  constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
  template<typename Arg, std::enable_if_t<(square_matrix<Arg, Likelihood::maybe>) and (max_tensor_order_of_v<Arg> <= 2), int> = 0>
  constexpr auto
#endif
  trace(Arg&& arg)
  {
    constexpr auto ix = []{ if constexpr (dynamic_dimension<Arg, 0>) return 1; else return 0; }();

    if constexpr (identity_matrix<Arg>)
    {
      return internal::index_dimension_scalar_constant_of<ix>(arg);
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      std::multiplies<scalar_type_of_t<Arg>> op;
      return internal::scalar_constant_operation{op, constant_diagonal_coefficient{arg}, internal::index_dimension_scalar_constant_of<ix>(arg)};
    }
    else if constexpr (dimension_size_of_index_is<Arg, 0, 1> or dimension_size_of_index_is<Arg, 1, 1>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'trace' is not a square matrix"};
      return constant_coefficient {arg};
    }
    else if constexpr (zero_matrix<Arg> or dimension_size_of_index_is<Arg, 0, 0> or dimension_size_of_index_is<Arg, 1, 0>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'trace' is not a square matrix"};
      return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0>{};
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'trace' is not a square matrix"};
      std::multiplies<scalar_type_of_t<Arg>> op;
      return internal::scalar_constant_operation{op, constant_coefficient{arg}, internal::index_dimension_scalar_constant_of<ix>(arg)};
    }
    else if constexpr (triangular_matrix<Arg> and not dynamic_dimension<Arg, ix> and index_dimension_of_v<Arg, ix> >= 2) // this includes the diagonal case
    {
      return reduce(std::plus<scalar_type_of_t<Arg>>{}, diagonal_of(std::forward<Arg>(arg)));
    }
    else
    {
      auto diag = diagonal_of(std::forward<Arg>(arg));
      if constexpr(dynamic_dimension<decltype(diag), 0>)
      {
        auto dim = get_index_dimension_of<0>(diag);
        if (dim >= 2) return static_cast<scalar_type_of_t<Arg>>(reduce(std::plus<scalar_type_of_t<Arg>>{}, std::move(diag)));
        else if (dim == 1) return static_cast<scalar_type_of_t<Arg>>(constant_coefficient {std::move(diag)});
        else return static_cast<scalar_type_of_t<Arg>>(0);
      }
      else
      {
        // diag is known at compile time to have at least 2 dimensions
        return reduce(std::plus<scalar_type_of_t<Arg>>{}, std::move(diag));
      }
    }
  }


  namespace detail
  {
    template<typename C, typename A, typename B, std::size_t...Is>
    static constexpr decltype(auto) contract_constant(C&& c, A&& a, B&& b, std::index_sequence<Is...>) noexcept
    {
      return make_constant_matrix_like<A>(std::forward<C>(c),
        get_vector_space_descriptor<0>(a), get_vector_space_descriptor<1>(b), get_vector_space_descriptor<Is + 2>(a)...);
    }


    template<std::size_t I, typename T, typename...Ts>
    constexpr decltype(auto) best_vector_space_descriptor(T&& t, Ts&&...ts)
    {
       if constexpr (sizeof...(Ts) == 0 or not dynamic_dimension<T, I>) return get_vector_space_descriptor<I>(t);
       else return best_vector_space_descriptor<I>(std::forward<Ts>(ts)...);
    }


    template<std::size_t...I, typename T>
    constexpr decltype(auto) sum_impl(std::index_sequence<I...>, T&& t) { return std::forward<T>(t); }


    template<std::size_t...I, typename T0, typename T1, typename...Ts>
    constexpr decltype(auto) sum_impl(std::index_sequence<I...> seq, T0&& t0, T1&& t1, Ts&&...ts)
    {
      if constexpr ((zero_matrix<T0> or zero_matrix<T1> or (constant_matrix<T0> and constant_matrix<T1>)) and not vector_space_descriptor_match<T0, T1>)
      {
        if (not get_vector_space_descriptor_match(t0, t1))
          throw std::invalid_argument {"In sum function, vector space descriptors of arguments do not match"};
      }

      if constexpr (zero_matrix<T0>)
      {
        return sum_impl(seq, std::forward<T1>(t1), std::forward<Ts>(ts)...);
      }
      else if constexpr (zero_matrix<T1>)
      {
        return sum_impl(seq, std::forward<T0>(t0), std::forward<Ts>(ts)...);
      }
      else if constexpr ((constant_matrix<T0> and constant_matrix<T1>))
      {
        auto c = constant_coefficient{t0} + constant_coefficient{t1};
        auto cm = make_constant_matrix_like<T0>(std::move(c), best_vector_space_descriptor<I>(t0, t1, ts...)...);
        auto ret = sum_impl(seq, std::move(cm), std::forward<Ts>(ts)...);
        return ret;
      }
      else if constexpr (constant_matrix<T0> and sizeof...(Ts) > 0) // Shift T0 right in hopes that it will combine with another constant
      {
        return sum_impl(seq, std::forward<T1>(t1), sum_impl(seq, std::forward<T0>(t0), std::forward<Ts>(ts)...));
      }
      else if constexpr (constant_matrix<T1> and sizeof...(Ts) > 0) // Shift T1 right in hopes that it will combine with another constant
      {
        return sum_impl(seq, std::forward<T0>(t0), sum_impl(seq, std::forward<T1>(t1), std::forward<Ts>(ts)...));
      }
      else if constexpr (diagonal_matrix<T0> and diagonal_matrix<T1>)
      {
        auto ret = sum_impl(seq, to_diagonal(sum_impl(seq, diagonal_of(std::forward<T0>(t0)), diagonal_of(std::forward<T1>(t1)))), std::forward<Ts>(ts)...);
        return ret;
      }
      else
      {
        auto ret = sum_impl(seq, interface::library_interface<std::decay_t<T0>>::sum(std::forward<T0>(t0), std::forward<T1>(t1)), std::forward<Ts>(ts)...);
        return ret;
      }
    }
  } // namespace detail


  /**
   * \brief Element-by-element sum of one or more objects.
   */
#ifdef __cpp_concepts
  template<indexible...Ts> requires (sizeof...(Ts) > 0) and maybe_vector_space_descriptor_match<Ts...>
#else
  template<typename...Ts, std::enable_if_t<(indexible<Ts> and ...) and (sizeof...(Ts) > 0) and
    maybe_vector_space_descriptor_match<Ts...>, int> = 0>
#endif
  constexpr decltype(auto) sum(Ts&&...ts)
  {
    // \todo Create a new wrapper argument that uses best_vector_space_descriptor above and guarantees a set of compile-time dimensions.
    constexpr std::make_index_sequence<std::max({index_count_v<Ts>...})> seq;

    if constexpr ((... and constant_matrix<Ts>))
    {
      return detail::sum_impl(seq, std::forward<Ts>(ts)...);
    }
    else if constexpr ((... and diagonal_matrix<Ts>))
    {
      return to_diagonal(detail::sum_impl(seq, diagonal_of(std::forward<Ts>(ts))...));
    }
    else if constexpr (triangle_type_of_v<Ts...> != TriangleType::any)
    {
      constexpr auto t = triangle_type_of_v<Ts...>;
      auto f = [](auto&& a) -> decltype(auto) {
        if constexpr (triangular_adapter<decltype(a)>) return nested_matrix(std::forward<decltype(a)>(a));
        else return std::forward<decltype(a)>(a);
      };
      return make_triangular_matrix<t>(detail::sum_impl(seq, f(std::forward<Ts>(ts))...));
    }
    else if constexpr ((... and hermitian_matrix<Ts>))
    {
      constexpr auto t = hermitian_adapter_type_of_v<Ts...> == HermitianAdapterType::any ?
        HermitianAdapterType::lower : hermitian_adapter_type_of_v<Ts...>;
      auto f = [](auto&& a) -> decltype(auto) {
        static_assert(hermitian_adapter_type_of_v<decltype(a)> != HermitianAdapterType::any);
        if constexpr (hermitian_adapter_type_of_v<decltype(a)> == t) return nested_matrix(std::forward<decltype(a)>(a));
        else if constexpr (hermitian_adapter<decltype(a)>) return transpose(nested_matrix(std::forward<decltype(a)>(a)));
        else return std::forward<decltype(a)>(a);
      };
      return make_hermitian_matrix<t>(detail::sum_impl(seq, f(std::forward<Ts>(ts))...));
    }
    else
    {
      return detail::sum_impl(seq, std::forward<Ts>(ts)...);
    }
  }


  /* // Only for use with alternate code below
  namespace detail
  {
    template<typename A, typename B, std::size_t...Is>
    static constexpr auto contract_dimensions(A&& a, B&& b, std::index_sequence<Is...>) noexcept
    {
      return std::tuple {get_vector_space_descriptor<0>(a), get_vector_space_descriptor<1>(b), get_vector_space_descriptor<Is + 2>(a)...};
    }
  }*/


  /**
   * \brief Matrix multiplication of A * B.
   */
#ifdef __cpp_concepts
  template<indexible A, indexible B> requires dimension_size_of_index_is<A, 1, index_dimension_of_v<B, 0>, Likelihood::maybe>
#else
  template<typename A, typename B, std::enable_if_t<indexible<A> and indexible<B> and
    (dimension_size_of_index_is<A, 1, index_dimension_of<B, 0>::value, Likelihood::maybe>), int> = 0>
#endif
  constexpr decltype(auto) contract(A&& a, B&& b)
  {
    if constexpr (dynamic_dimension<A, 1> or dynamic_dimension<B, 0>) if (get_vector_space_descriptor<1>(a) != get_vector_space_descriptor<0>(b))
      throw std::domain_error {"In contract, columns of a (" + std::to_string(get_index_dimension_of<1>(a)) +
        ") do not match rows of b (" + std::to_string(get_index_dimension_of<0>(b)) + ")"};

    constexpr std::size_t dims = std::max({index_count_v<A>, index_count_v<B>, static_cast<std::size_t>(2)});
    constexpr std::make_index_sequence<dims - 2> seq;

    if constexpr (identity_matrix<B>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (identity_matrix<A>)
    {
      return std::forward<B>(b);
    }
    else if constexpr (zero_matrix<A> or zero_matrix<B>)
    {
      using Scalar = std::decay_t<decltype(std::declval<scalar_type_of_t<A>>() * std::declval<scalar_type_of_t<B>>())>;
      return detail::contract_constant(internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{}, std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (constant_matrix<A> and constant_matrix<B>)
    {
      auto dim_const = [](const auto& a, const auto& b) {
        if constexpr (dynamic_dimension<A, 1>) return internal::index_dimension_scalar_constant_of<0>(b);
        else return internal::index_dimension_scalar_constant_of<1>(a);
      }(a, b);

      auto abd = constant_coefficient{a} * constant_coefficient{b} * std::move(dim_const);
      return detail::contract_constant(std::move(abd), std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (diagonal_matrix<A> and constant_matrix<B>)
    {
      auto col = make_self_contained(diagonal_of(std::forward<A>(a)) * constant_coefficient{b}());
      return chipwise_operation<1>([&]{ return col; }, get_index_dimension_of<1>(b));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, make_self_contained(std::move(col)));
    }
    else if constexpr (constant_matrix<A> and diagonal_matrix<B>)
    {
      auto row = make_self_contained(transpose(diagonal_of(std::forward<B>(b))) * constant_coefficient{a}());
      return chipwise_operation<0>([&]{ return row; }, get_index_dimension_of<0>(a));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, make_self_contained(std::move(row)));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<B>)
    {
      auto ret = to_diagonal(n_ary_operation(std::multiplies<>{}, diagonal_of(std::forward<A>(a)), diagonal_of(std::forward<B>(b))));
      return ret;
    }
    else
    {
      return interface::library_interface<std::decay_t<A>>::contract(std::forward<A>(a), std::forward<B>(b));
    }
  }


#ifndef __cpp_concepts
    namespace detail
    {
      template<bool on_the_right, typename A, typename B, typename = void>
      struct contract_in_place_exists : std::false_type {};

      template<bool on_the_right, typename A, typename B>
      struct contract_in_place_exists<on_the_right, A, B, std::void_t<decltype(
        interface::library_interface<std::decay_t<A>>::template contract<on_the_right>(std::declval<A&>(), std::declval<B&&>()))>>
        : std::true_type {};
    }
#endif


  /**
   * \brief In-place matrix multiplication of A * B, storing the result in A
   * \tparam on_the_right Whether the application is on the right (true) or on the left (false)
   * \result Either either A * B (if on_the_right == true) or B * A (if on_the_right == false)
   */
#ifdef __cpp_concepts
  template<bool on_the_right = true, square_matrix<Likelihood::maybe> A, square_matrix<Likelihood::maybe> B> requires
    maybe_has_same_shape_as<A, B> and (writable<A> or triangle_type_of_v<A> == triangle_type_of_v<A, B>)
#else
  template<bool on_the_right = true, typename A, typename B, std::enable_if_t<
    square_matrix<A, Likelihood::maybe> and square_matrix<B, Likelihood::maybe> and maybe_has_same_shape_as<A, B> and
    (writable<A> or triangle_type_of_v<A> == triangle_type_of_v<A, B>), int> = 0>
#endif
  constexpr A& contract_in_place(A& a, B&& b)
  {
    if constexpr (not square_matrix<A> or not square_matrix<B> or not has_same_shape_as<A, B>) if (not get_vector_space_descriptor_match(a, b))
      throw std::invalid_argument {"Arguments to contract_in_place must match in size and be square matrices"};

    if constexpr (zero_matrix<A> or identity_matrix<B>)
    {
      return a;
    }
    else if constexpr (zero_matrix<B>)
    {
      return a = std::forward<B>(b);
    }
    else if constexpr (diagonal_adapter<A> and diagonal_matrix<B>)
    {
      internal::set_triangle<TriangleType::diagonal>(a, n_ary_operation(std::multiplies<>{}, diagonal_of(a), diagonal_of(std::forward<B>(b))));
      return a;
    }
    else if constexpr (triangular_adapter<A> and triangle_type_of_v<A> == triangle_type_of_v<A, B>)
    {
      internal::set_triangle<triangle_type_of_v<A>>(a, contract(a, std::forward<B>(b)));
      return a;
    }
#ifdef __cpp_concepts
    else if constexpr (requires { interface::library_interface<std::decay_t<A>>::template contract_in_place<on_the_right>(a, std::forward<B>(b)); })
#else
    else if constexpr (detail::contract_in_place_exists<on_the_right, A, B>::value)
#endif
    {
      return interface::library_interface<std::decay_t<A>>::template contract_in_place<on_the_right>(a, std::forward<B>(b));
    }
    else
    {
      a = contract(a, std::forward<B>(b));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP
