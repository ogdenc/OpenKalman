/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to native Eigen types.
 */

#ifndef OPENKALMAN_EIGEN3_GENERAL_TRAITS_HPP
#define OPENKALMAN_EIGEN3_GENERAL_TRAITS_HPP

#include <type_traits>
#include <tuple>
#include <random>


namespace OpenKalman::Eigen3
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct eigen_has_linear_access : std::false_type {};

    template<typename T>
    struct eigen_has_linear_access<T, std::enable_if_t<Eigen3::native_eigen_dense<T>>>
      : std::bool_constant<(Eigen::internal::evaluator<T>::Flags & Eigen::LinearAccessBit) != 0> {};
  }
#endif


#ifdef __cpp_concepts
  template<Eigen3::native_eigen_dense T>
  struct IndexibleObjectTraitsBase<T>
#else
  template<typename T>
  struct IndexibleObjectTraitsBase<T, std::enable_if_t<Eigen3::native_eigen_dense<T> and (not Eigen3::eigen_wrapper<T>)>>
#endif
  {
    using scalar_type = typename std::decay_t<T>::Scalar;


#ifdef __cpp_lib_concepts
    template<typename Arg, typename...I> requires (sizeof...(I) > 0) and (sizeof...(I) <= 2) and
      (std::convertible_to<I, Eigen::Index> and ...) and
      (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<Arg>>::Flags & Eigen::LinearAccessBit) != 0)
#else
    template<typename Arg, typename...I, std::enable_if_t<(sizeof...(I) > 0) and (sizeof...(I) <= 2) and
      (std::is_convertible_v<I, Eigen::Index> and ...) and
      (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<Arg>>::Flags & Eigen::LinearAccessBit) != 0) , int> = 0>
#endif
    static constexpr decltype(auto) get(Arg&& arg, I...i)
    {
      if constexpr ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0)
        return std::forward<Arg>(arg).coeffRef(static_cast<Eigen::Index>(i)...);
      else
        return std::forward<Arg>(arg).coeff(static_cast<Eigen::Index>(i)...);
    }


#ifdef __cpp_lib_concepts
    template<typename Arg, typename...I> requires (sizeof...(I) > 0) and (sizeof...(I) <= 2) and
      (std::convertible_to<I, Eigen::Index> and ...) and
      (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<Arg>>::Flags & Eigen::LinearAccessBit) != 0) and
      ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0)
#else
    template<typename Arg, typename...I, std::enable_if_t<(sizeof...(I) > 0) and (sizeof...(I) <= 2) and
      (std::is_convertible_v<I, Eigen::Index> and ...) and
      (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<Arg>>::Flags & Eigen::LinearAccessBit) != 0) and
      ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0), int> = 0>
#endif
    static void set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
    {
      arg.coeffRef(i...) = s;
    }


    static constexpr bool is_writable =
      static_cast<bool>(Eigen::internal::traits<std::decay_t<T>>::Flags & (Eigen::LvalueBit | Eigen::DirectAccessBit));
  };


#ifdef __cpp_concepts
  template<typename T> requires Eigen3::eigen_DiagonalMatrix<T> or Eigen3::eigen_DiagonalWrapper<T>
  struct IndexibleObjectTraitsBase<T>
#else
  template<typename T>
  struct IndexibleObjectTraitsBase<T, std::enable_if_t<Eigen3::eigen_DiagonalMatrix<T> or Eigen3::eigen_DiagonalWrapper<T>>>
#endif
  {
    using scalar_type = typename std::decay_t<T>::Scalar;


#ifdef __cpp_concepts
    template<typename Arg> requires element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 1>
#else
    template<typename Arg, std::enable_if_t<element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 1>, int> = 0>
#endif
    static scalar_type_of_t<Arg> get(const Arg& arg, Eigen::Index i)
    {
      return get_element(nested_matrix(arg), i);
    }


#ifdef __cpp_concepts
    template<typename Arg> requires element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 1> or
      element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 2>
#else
    template<typename Arg, std::enable_if_t<element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 1> or
      element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 2>, int> = 0>
#endif
    static constexpr auto get(const Arg& arg, Eigen::Index i, Eigen::Index j)
    {
      if (i == j)
      {
        if constexpr (element_gettable<nested_matrix_of_t<const Arg&>, 1>)
          return get_element(nested_matrix(arg), i);
        else
          return get_element(nested_matrix(arg), i, i);
      }
      else
      {
        return scalar_type_of_t<Arg>(0);
      }
    }


#ifdef __cpp_concepts
    template<typename Arg> requires element_settable<decltype(nested_matrix(std::declval<Arg&>())), 1> or
      element_settable<nested_matrix_of_t<Arg&>, 2>
#else
    template<typename Arg, std::enable_if_t<element_settable<decltype(nested_matrix(std::declval<Arg&>())), 1> or
      element_settable<typename nested_matrix_of<Arg&>::type, 2>, int> = 0>
#endif
    static void set(Arg& arg, const scalar_type_of_t<Arg>& s, Eigen::Index i)
    {
      if constexpr (element_settable<nested_matrix_of_t<Arg&>, 1>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, i);
    }


#ifdef __cpp_concepts
    template<typename Arg> requires element_settable<decltype(nested_matrix(std::declval<Arg&>())), 1> or
      element_settable<decltype(nested_matrix(std::declval<Arg&>())), 2>
#else
    template<typename Arg, std::enable_if_t<element_settable<decltype(nested_matrix(std::declval<Arg&>())), 1> or
      element_settable<decltype(nested_matrix(std::declval<Arg&>())), 2>, int> = 0>
#endif
    static void set(Arg& arg, const scalar_type_of_t<Arg>& s, Eigen::Index i, Eigen::Index j)
    {
      if (i == j or s == 0)
      {
        if constexpr (element_settable<nested_matrix_of_t<Arg&>, 1>)
          set_element(nested_matrix(arg), s, i);
        else
          set_element(nested_matrix(arg), s, i, i);
      }
      else throw std::invalid_argument {"Off-diagonal elements of DiagonalMatrix or DiagonalWrapper can only be set to zero"};
    }

    static constexpr bool is_writable = false;

  };

} // namespace OpenKalman::interface


namespace OpenKalman::interface
{
#ifdef __cpp_concepts
  template<Eigen3::native_eigen_general T>
  struct LibraryRoutines<T>
#else
  template<typename T>
  struct LibraryRoutines<T, std::enable_if_t<Eigen3::native_eigen_general<T>>>
#endif
  {
    template<typename Derived>
    using LibraryBase = Eigen3::Eigen3AdapterBase<Derived, T>;

  private:

    template<typename Scalar, Eigen::Index...Is>
    using dense_type = std::conditional_t<Eigen3::native_eigen_array<T>,
      Eigen::Array<Scalar, Is...>, Eigen::Matrix<Scalar, Is...>>;

    template<typename Scalar, std::size_t...Is>
    using writable_type = dense_type<Scalar, (Is == dynamic_size ? Eigen::Dynamic : static_cast<Eigen::Index>(Is))...>;

#  ifdef __cpp_concepts
    template<typename ArgType>
#  else
    template<typename ArgType, typename = void>
#  endif
    struct traits_and_evaluator_defined : std::false_type {};


#  ifdef __cpp_concepts
     template<typename ArgType> requires
       requires { typename Eigen::internal::traits<ArgType>; typename Eigen::internal::evaluator<ArgType>; }
     struct traits_and_evaluator_defined<ArgType> : std::true_type {};
#  else
     template<typename ArgType>
     struct traits_and_evaluator_defined<ArgType, std::enable_if_t<
       std::is_void<std::void_t<Eigen::internal::traits<ArgType>>>::value and
       std::is_void<std::void_t<Eigen::internal::evaluator<ArgType>>>::value>> : std::true_type {};
#  endif

  public:

    template<typename Scalar, typename...D>
    static auto make_default(D&&...d)
    {
      using M = writable_type<Scalar, dimension_size_of_v<D>...>;

      if constexpr (((dimension_size_of_v<D> == dynamic_size) or ...))
        return M(static_cast<Eigen::Index>(get_dimension_size_of(d))...);
      else
        return M {};
    }


    template<typename Scalar, typename Arg>
    static decltype(auto) convert(Arg&& arg)
    {
      using M = writable_type<Scalar, index_dimension_of_v<Arg, 0>, index_dimension_of_v<Arg, 1>>;

      if constexpr (std::is_base_of_v<Eigen::PlainObjectBase<std::decay_t<Arg>>, std::decay_t<Arg>> and
        not std::is_const_v<std::remove_reference_t<Arg>>)
      {
        return std::forward<Arg>(arg);
      }
      else if constexpr (std::is_constructible_v<M, Arg&&>)
      {
        return M {std::forward<Arg>(arg)};
      }
      else
      {
        auto r = get_index_dimension_of<0>(arg);
        auto c = get_index_dimension_of<1>(arg);
        auto m = make_default<Scalar>(r, c);
        for (int i = 0; i < r ; ++i) for (int j = 0; j < c ; ++j)
        {
          set_element(m, get_element(std::forward<Arg>(arg), i, j), i, j);
        }
        return m;
      }
    }


    template<typename Arg>
    static decltype(auto) to_native_matrix(Arg&& arg)
    {
      if constexpr (Eigen3::native_eigen_matrix<Arg>)
        return std::forward<Arg>(arg);
      else if constexpr (Eigen3::native_eigen_array<Arg>)
        return std::forward<Arg>(arg).matrix();
      else if constexpr (Eigen3::native_eigen_general<Arg>)
        return convert<scalar_type_of_t<Arg>>(std::forward<Arg>(arg));
      else
      {
        static_assert(traits_and_evaluator_defined<Arg>::value, "To convert to a native Eigen matrix, the interface "
          "must define a trait and an evaluator for the argument");
        return Eigen3::EigenWrapper<std::decay_t<Arg>> {std::forward<Arg>(arg)};
      }
    }


    template<typename Arg>
    static decltype(auto) to_native_matrix(const Eigen3::EigenWrapper<Arg>& arg) { return arg; }


    template<typename Arg>
    static decltype(auto) to_native_matrix(Eigen3::EigenWrapper<Arg>&& arg) { return std::move(arg); }


#ifdef __cpp_concepts
    template<scalar_type Scalar, index_descriptor Drow, index_descriptor Dcol, std::convertible_to<Scalar> Arg, std::convertible_to<Scalar> ... Args>
#else
    template<typename Scalar, typename Drow, typename Dcol, typename Arg, typename...Args, std::enable_if_t<
      scalar_type<Scalar> and index_descriptor<Drow> and index_descriptor<Dcol> and
      std::conjunction_v<std::is_convertible<Arg, Scalar>, std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto make_from_elements(const std::tuple<Drow, Dcol>& d_tup, const Arg arg, const Args ... args)
    {
      constexpr std::size_t rows = dimension_size_of_v<Drow>;
      constexpr std::size_t columns = dimension_size_of_v<Dcol>;
      if constexpr (rows != dynamic_size and columns != dynamic_size)
        return ((Eigen3::eigen_matrix_t<Scalar, rows, columns> {} << arg), ... , args).finished();
      else if constexpr (rows != dynamic_size and columns == dynamic_size)
        return ((Eigen3::eigen_matrix_t<Scalar, rows, (1 + sizeof...(Args)) / rows> {} << arg), ... , args).finished();
      else if constexpr (rows == dynamic_size and columns != dynamic_size)
        return ((Eigen3::eigen_matrix_t<Scalar, (1 + sizeof...(Args)) / columns, columns> {} << arg), ... , args).finished();
      else
      {
        static_assert(rows == dynamic_size and columns == dynamic_size);
        return ((Eigen3::eigen_matrix_t<Scalar, 1 + sizeof...(Args), 1> {} << arg), ... , args).finished();
      }
    }


#ifdef __cpp_concepts
    template<scalar_constant<CompileTimeStatus::unknown> C, typename...Ds> requires (sizeof...(Ds) == 2)
    static constexpr constant_matrix<CompileTimeStatus::unknown> auto
#else
    template<typename C, typename...Ds, std::enable_if_t<
      scalar_constant<C, CompileTimeStatus::unknown> and (sizeof...(Ds) == 2), int> = 0>
    static constexpr auto
#endif
    make_constant_matrix(C&& c, Ds&&...ds)
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      using N = dense_writable_matrix_t<T, Scalar, std::decay_t<Ds>...>;
      return N::Constant(static_cast<Eigen::Index>(get_dimension_size_of(std::forward<Ds>(ds)))..., std::forward<C>(c));
    }


    template<typename Scalar, typename D>
    static constexpr auto
    make_identity_matrix(D&& d)
    {
      if constexpr (dimension_size_of_v<D> == dynamic_size)
      {
        return to_diagonal(make_constant_matrix_like<T, Scalar, 1>(std::forward<D>(d), Dimensions<1>{}));
      }
      else
      {
        constexpr Eigen::Index n {dimension_size_of_v<D>};
        return Eigen3::eigen_matrix_t<Scalar, n, n>::Identity();
      }
    }


    template<typename Arg, typename...Begin, typename...Size>
    static auto
    get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      static_assert(sizeof...(Begin) == 2 and sizeof...(Size) == 2);

      if constexpr (Eigen3::native_eigen_dense<Arg>)
      {
        using B = Eigen::Block<std::remove_reference_t<Arg>,
          static_cast<Eigen::Index>(static_index_value<Size> ? static_index_value_of_v<Size> : Eigen::Dynamic)...>;

        if constexpr ((static_index_value<Size> and ...))
          return make_self_contained<Arg>(B(arg, std::get<0>(begin), std::get<1>(begin)));
        else
          return make_self_contained<Arg>(B(arg, std::get<0>(begin), std::get<1>(begin), std::get<0>(size), std::get<1>(size)));
      }
      else return make_self_contained(get_block(Eigen3::EigenWrapper {std::forward<Arg>(arg)}, begin, size));
    }


    template<typename Arg, typename Block, typename...Begin>
    static constexpr Arg&
    set_block(Arg& arg, Block&& block, Begin...begin)
    {
      static_assert(sizeof...(Begin) == 2);
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        set_block(nested_matrix(arg), std::forward<Block>(block), begin...);
        return arg;
      }
      else
      {
        static_assert(Eigen3::native_eigen_dense<Arg>);

        if constexpr (Eigen3::eigen_block<Block>)
        {
          if (std::addressof(arg) == std::addressof(block.nestedExpression()) and
              std::get<0>(std::tuple{begin...}) == block.startRow() and std::get<1>(std::tuple{begin...}) == block.startCol())
            return arg;
        }

        using B = Eigen::Block<std::remove_reference_t<Arg>,
          static_cast<Eigen::Index>(index_dimension_of_v<Block, 0>),
          static_cast<Eigen::Index>(index_dimension_of_v<Block, 1>)>;

        if constexpr (not has_dynamic_dimensions<Block>)
          B(arg, begin...) = std::forward<Block>(block);
        else
          B(arg, begin..., get_index_dimension_of<0>(block), get_index_dimension_of<1>(block)) = std::forward<Block>(block);
        return arg;
      }
    }

  private:

#ifdef __cpp_concepts
    template<typename A>
#else
    template<typename A, typename = void>
#endif
    struct pass_through_eigenwrapper : std::false_type {};

#ifdef __cpp_concepts
    template<Eigen3::eigen_wrapper A>
    struct pass_through_eigenwrapper<A>
#else
    template<typename A>
    struct pass_through_eigenwrapper<A, std::enable_if_t<Eigen3::eigen_wrapper<A>>>
#endif
      : std::bool_constant<Eigen3::native_eigen_dense<nested_matrix_of_t<A>> or diagonal_adapter<nested_matrix_of_t<A>> or
        triangular_adapter<nested_matrix_of_t<A>> or hermitian_adapter<nested_matrix_of_t<A>>> {};

  public:

    template<TriangleType t, typename A, typename B>
    static decltype(auto) set_triangle(A&& a, B&& b)
    {
      if constexpr (Eigen3::eigen_MatrixWrapper<A> or Eigen3::eigen_ArrayWrapper<A> or pass_through_eigenwrapper<A>::value)
      {
        return internal::set_triangle<t>(nested_matrix(std::forward<A>(a)), std::forward<B>(b));
      }
      else if constexpr (not Eigen3::native_eigen_dense<A>)
      {
        return set_triangle<t>(Eigen3::EigenWrapper {std::forward<A>(a)}, std::forward<B>(b));
      }
      else
      {
        if constexpr (t == TriangleType::diagonal)
        {
          if constexpr (writable<A> and std::is_lvalue_reference_v<A>)
          {
            a.diagonal() = OpenKalman::diagonal_of(std::forward<B>(b));
            return std::forward<A>(a);
          }
          else
          {
            auto aw = make_dense_writable_matrix_from(std::forward<A>(a));
            aw.diagonal() = OpenKalman::diagonal_of(std::forward<B>(b));
            return aw;
          }
        }
        else
        {
          decltype(auto) aw = [](A&& a) -> decltype(auto) {
            if constexpr (writable<A> and std::is_lvalue_reference_v<A>) return std::forward<A>(a);
            else return make_dense_writable_matrix_from(std::forward<A>(a));
          }(std::forward<A>(a));

          auto tview = aw.template triangularView<t == TriangleType::upper ? Eigen::Upper : Eigen::Lower>();
          if constexpr (std::is_assignable_v<decltype(tview), B&&>)
            tview = std::forward<B>(b);
          else
            tview = Eigen3::EigenWrapper{std::forward<B>(b)};
          return std::forward<decltype(aw)>(aw);
        }
      }
    }


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_dense Arg> requires
      std::is_lvalue_reference_v<nested_matrix_of_t<Eigen::DiagonalWrapper<std::remove_reference_t<Arg>>>>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_dense<Arg> and
      std::is_lvalue_reference_v<typename nested_matrix_of<Eigen::DiagonalWrapper<std::remove_reference_t<Arg>>>::type>, int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg& arg)
    {
      if constexpr (not vector<Arg>) if (not get_is_vector(arg)) throw std::invalid_argument {
        "Argument of to_diagonal must have 1 column; instead it has " + std::to_string(get_index_dimension_of<1>(arg))};
      return Eigen::DiagonalWrapper<std::remove_reference_t<Arg>> {arg};
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_SelfAdjointView Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_SelfAdjointView<Arg>, int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      // If it is a column vector, the SelfAdjointView wrapper doesn't matter, and otherwise, the following will throw an exception:
      return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_TriangularView Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_TriangularView<Arg>, int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      // If it is a column vector, the TriangularView wrapper doesn't matter, and otherwise, the following will thow an exception:
      return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg)));
    }

  private:

    template<typename Arg, typename Id>
    static constexpr decltype(auto)
    diagonal_of_impl(Arg&& arg, Id&& id)
    {
      auto diag {[](Arg&& arg){
        if constexpr (Eigen3::native_eigen_array<Arg>)
          return make_self_contained<Arg>(std::forward<Arg>(arg).matrix().diagonal());
        else // native_eigen_matrix<Arg>
          return make_self_contained<Arg>(std::forward<Arg>(arg).diagonal());
      }(std::forward<Arg>(arg))};
      using Diag = const std::decay_t<decltype(diag)>;
      static_assert(vector<Diag>);

      using D = std::conditional_t<dynamic_dimension<Diag, 0> and fixed_index_descriptor<Id>, std::decay_t<Id>, index_descriptor_of_t<Diag, 0>>;

      if constexpr (not dynamic_dimension<Diag, 0> or dynamic_index_descriptor<D>) return diag;
      else return internal::FixedSizeAdapter<Diag, D, Dimensions<1>> {std::move(diag)};
    }

  public:

#ifdef __cpp_concepts
    template<Eigen3::native_eigen_general Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_general<Arg>, int> = 0>
#endif
    static constexpr decltype(auto)
    diagonal_of(Arg&& arg)
    {
      auto d = get_is_square(arg);
      if (not d) throw std::invalid_argument {"Argument of diagonal_of must be a square matrix; instead it has " +
          std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
          std::to_string(get_index_dimension_of<1>(arg)) + " columns"};

      using Scalar = scalar_type_of_t<Arg>;

      if constexpr (Eigen3::eigen_DiagonalWrapper<Arg>)
      {
        using Scalar = scalar_type_of_t<Arg>;
        decltype(auto) diag {nested_matrix(std::forward<Arg>(arg))}; //< must be nested_matrix(...) rather than .diagonal() because of const_cast
        using Diag = decltype(diag);
        using EigenTraits = Eigen::internal::traits<std::decay_t<Diag>>;
        constexpr Eigen::Index rows = EigenTraits::RowsAtCompileTime;
        constexpr Eigen::Index cols = EigenTraits::ColsAtCompileTime;

        static_assert(cols != 1, "For Eigen::DiagonalWrapper<T> interface, T should never be a column vector "
                                 "because diagonal_of function handles this case.");
        if constexpr (cols == 0)
        {
          return std::forward<Diag>(diag);
        }
        else if constexpr (rows == 1 or rows == 0)
        {
          return OpenKalman::transpose(std::forward<Diag>(diag));
        }
        else if constexpr (rows == Eigen::Dynamic or cols == Eigen::Dynamic)
        {
          auto d {make_dense_writable_matrix_from(std::forward<Diag>(diag))};
          using M = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
          return M {M::Map(make_dense_writable_matrix_from(std::forward<Diag>(diag)).data(),
            get_index_dimension_of<0>(diag) * get_index_dimension_of<1>(diag))};
        }
        else // rows > 1 and cols > 1
        {
          using M = Eigen::Matrix<Scalar, rows * cols, 1>;
          return M {M::Map(make_dense_writable_matrix_from(std::forward<Diag>(diag)).data())};
        }
      }
      else if constexpr (Eigen3::eigen_SelfAdjointView<Arg> or Eigen3::eigen_TriangularView<Arg>)
      {
        // Assume there are no dangling references
        return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::native_eigen_general<nested_matrix_of_t<Arg>>)
          return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg)));
        else
          return diagonal_of_impl(std::forward<Arg>(arg), *d);
      }
      else if constexpr (Eigen3::eigen_Identity<Arg>)
      {
        constexpr std::size_t dim = dynamic_rows<Arg> ? column_dimension_of_v<Arg> : row_dimension_of_v<Arg>;
        if constexpr (dim == dynamic_size) return make_constant_matrix_like<Arg, Scalar, 1>(*d, Dimensions<1>{});
        else return make_constant_matrix_like<Arg, Scalar, 1>(Dimensions<dim>{}, Dimensions<1>{});
      }
      else
      {
        return diagonal_of_impl(std::forward<Arg>(arg), *d);
      };
    }

  private:

    // Convert std functions to equivalent Eigen operations for possible vectorization:
    template<typename Op> static decltype(auto) nat_op(Op&& op) { return std::forward<Op>(op); };
    template<typename S> static auto nat_op(const std::plus<S>& op) { return Eigen::internal::scalar_sum_op<S, S> {}; };
    template<typename S> static auto nat_op(const std::minus<S>& op) { return Eigen::internal::scalar_difference_op<S, S> {}; };
    template<typename S> static auto nat_op(const std::multiplies<S>& op) {return Eigen::internal::scalar_product_op<S, S> {}; };
    template<typename S> static auto nat_op(const std::divides<S>& op) { return Eigen::internal::scalar_quotient_op<S, S> {}; };
    template<typename S> static auto nat_op(const std::negate<S>& op) { return Eigen::internal::scalar_opposite_op<S> {}; };

    using EIC = Eigen::internal::ComparisonName;
    template<typename S> static auto nat_op(const std::equal_to<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_EQ> {}; };
    template<typename S> static auto nat_op(const std::not_equal_to<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_NEQ> {}; };
    template<typename S> static auto nat_op(const std::greater<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_GT> {}; };
    template<typename S> static auto nat_op(const std::less<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_LT> {}; };
    template<typename S> static auto nat_op(const std::greater_equal<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_GE> {}; };
    template<typename S> static auto nat_op(const std::less_equal<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_LE> {}; };

    template<typename S> static auto nat_op(const std::logical_and<S>& op) { return Eigen::internal::scalar_boolean_and_op {}; };
    template<typename S> static auto nat_op(const std::logical_or<S>& op) { return Eigen::internal::scalar_boolean_or_op {}; };
    template<typename S> static auto nat_op(const std::logical_not<S>& op) { return Eigen::internal::scalar_boolean_not_op<S> {}; };


    template<typename...Ds, typename...ArgDs, typename Arg, std::size_t...I>
    static decltype(auto)
    replicate_arg_impl(const std::tuple<Ds...>& p_tup, const std::tuple<ArgDs...>& arg_tup, Arg&& arg, std::index_sequence<I...>)
    {
      using R = Eigen::Replicate<std::decay_t<Arg>,
        (dimension_size_of_v<Ds> == dynamic_size or dimension_size_of_v<ArgDs> == dynamic_size ?
        Eigen::Dynamic : static_cast<Eigen::Index>(dimension_size_of_v<Ds> / dimension_size_of_v<ArgDs>))...>;

      if constexpr (((dimension_size_of_v<Ds> != dynamic_size) and ...) and
        ((dimension_size_of_v<ArgDs> != dynamic_size) and ...))
      {
        if constexpr (((dimension_size_of_v<Ds> == dimension_size_of_v<ArgDs>) and ...))
          return std::forward<Arg>(arg);
        else
          return R {std::forward<Arg>(arg)};
      }
      else
      {
        auto ret = R {std::forward<Arg>(arg), static_cast<Eigen::Index>(
          get_dimension_size_of(std::get<I>(p_tup)) / get_dimension_size_of(std::get<I>(arg_tup)))...};
        return ret;
      }
    }


    template<typename...Ds, typename Arg>
    static decltype(auto)
    replicate_arg(const std::tuple<Ds...>& p_tup, Arg&& arg)
    {
      return replicate_arg_impl(p_tup, get_all_dimensions_of(arg), std::forward<Arg>(arg), std::index_sequence_for<Ds...> {});
    }

  public:

#ifdef __cpp_concepts
    template<typename...Ds, typename Operation, typename...Args> requires (sizeof...(Args) <= 3) and
      std::invocable<Operation&&, scalar_type_of_t<Args>...> and
      scalar_type<std::invoke_result_t<Operation&&, scalar_type_of_t<Args>...>>
#else
    template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<(sizeof...(Args) <= 3) and
      std::is_invocable<Operation&&, typename scalar_type_of<Args>::type...>::value and
      scalar_type<typename std::invoke_result<Operation&&, typename scalar_type_of<Args>::type...>::type>, int> = 0>
#endif
    static auto
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& operation, Args&&...args)
    {
      decltype(auto) op = nat_op(std::forward<Operation>(operation));
      using Op = decltype(op);

      if constexpr (sizeof...(Args) == 0)
      {
        using P = dense_writable_matrix_t<T, scalar_type_of_t<T>, Ds...>;
        Eigen::Index r = get_dimension_size_of(std::get<0>(tup));
        Eigen::Index c = get_dimension_size_of(std::get<1>(tup));
        return Eigen::CwiseNullaryOp<std::decay_t<Op>, P> {r, c, std::forward<Op>(op)};
      }
      else if constexpr (sizeof...(Args) == 1)
      {
        return make_self_contained<Args...>(Eigen::CwiseUnaryOp {
          replicate_arg(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
      else if constexpr (sizeof...(Args) == 2)
      {
        return make_self_contained<Args...>(Eigen::CwiseBinaryOp<std::decay_t<Op>,
          std::decay_t<decltype(replicate_arg(tup, std::forward<Args>(args)))>...> {
          replicate_arg(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
      else
      {
        return make_self_contained<Args...>(Eigen::CwiseTernaryOp<std::decay_t<Op>,
          std::decay_t<decltype(replicate_arg(tup, std::forward<Args>(args)))>...> {
          replicate_arg(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
    }


#ifdef __cpp_concepts
    template<typename...Ds, typename Operation>
    requires std::invocable<Operation&&, typename dimension_size_of<Ds>::type...> and
      scalar_type<std::invoke_result_t<Operation&&, typename dimension_size_of<Ds>::type...>>
#else
    template<typename...Ds, typename Operation, std::enable_if_t<
      std::is_invocable<Operation&&, typename dimension_size_of<Ds>::type...>::value and
      scalar_type<typename std::invoke_result<Operation&&, typename dimension_size_of<Ds>::type...>::type>, int> = 0>
#endif
    static constexpr auto
    n_ary_operation_with_indices(const std::tuple<Ds...>& d_tup, Operation&& operation)
    {
      using P = dense_writable_matrix_t<T, scalar_type_of_t<T>, Ds...>;
      Eigen::Index r = get_dimension_size_of(std::get<0>(d_tup));
      Eigen::Index c = get_dimension_size_of(std::get<1>(d_tup));
      return Eigen::CwiseNullaryOp<std::decay_t<Operation>, P> {r, c, std::forward<Operation>(operation)};
    }


    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr auto
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      if constexpr (Eigen3::native_eigen_dense<Arg>)
      {
        decltype(auto) op = nat_op(std::forward<BinaryFunction>(b));
        using Op = decltype(op);

        if constexpr (sizeof...(indices) == 2) // reduce in both directions
        {
          return std::forward<Arg>(arg).redux(std::forward<Op>(op));
        }
        else
        {
          using OpWrapper = Eigen::internal::member_redux<std::decay_t<Op>, scalar_type_of_t<Arg>>;
          constexpr auto dir = ((indices == 0) and ...) ? Eigen::Vertical : Eigen::Horizontal;
          using P = Eigen::PartialReduxExpr<std::decay_t<Arg>, OpWrapper, dir>;
          return make_self_contained<Arg>(P {std::forward<Arg>(arg), OpWrapper {std::forward<Op>(op)}});
        }
      }
      else
      {
        return reduce<indices...>(std::forward<BinaryFunction>(b), Eigen3::EigenWrapper {std::forward<Arg>(arg)});
      }
    }

    // to_euclidean not defined--rely on default

    // from_euclidean not defined--rely on default

    // wrap_angles not defined--rely on default


    template<typename Arg>
    static constexpr decltype(auto)
    conjugate(Arg&& arg) noexcept
    {
      // The global conjugate function already handles DiagonalMatrix and DiagonalWrapper
      return std::forward<Arg>(arg).conjugate();
    }


    template<typename Arg>
    static constexpr decltype(auto)
    transpose(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::native_eigen_general<nested_matrix_of_t<Arg>>)
          return transpose(nested_matrix(std::forward<Arg>(arg)));
        else
          return std::forward<Arg>(arg).transpose(); // Rely on inherited Eigen transpose method
      }
      else if constexpr (Eigen3::native_eigen_matrix<Arg> or Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
        return std::forward<Arg>(arg).transpose();
      else if constexpr (Eigen3::native_eigen_array<Arg>)
        return std::forward<Arg>(arg).matrix().transpose();
      else if constexpr (triangular_matrix<Arg>)
        return OpenKalman::transpose(TriangularMatrix {std::forward<Arg>(arg)});
      else
        return Eigen3::EigenWrapper {std::forward<Arg>(arg)}.transpose();
      // Note: the global transpose function already handles zero, constant, constant-diagonal, and symmetric cases.
    }


    template<typename Arg>
    static constexpr decltype(auto)
    adjoint(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::native_eigen_general<nested_matrix_of_t<Arg>>)
          return adjoint(nested_matrix(std::forward<Arg>(arg)));
        else
          return std::forward<Arg>(arg).adjoint(); // Rely on inherited Eigen adjoint method
      }
      else if constexpr (Eigen3::native_eigen_matrix<Arg> or Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
        return std::forward<Arg>(arg).adjoint();
      else if constexpr (Eigen3::native_eigen_array<Arg>)
        return std::forward<Arg>(arg).matrix().adjoint();
      else if constexpr (triangular_matrix<Arg>)
        return OpenKalman::adjoint(TriangularMatrix {std::forward<Arg>(arg)});
      else
        return Eigen3::EigenWrapper {std::forward<Arg>(arg)}.adjoint();
      // Note: the global adjoint function already handles zero, constant, diagonal, non-complex, and hermitian cases.
    }


    template<typename Arg>
    static constexpr auto
    determinant(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::native_eigen_matrix<Arg>)
        return std::forward<Arg>(arg).determinant();
      else if constexpr (Eigen3::native_eigen_array<Arg>)
        return std::forward<Arg>(arg).matrix().determinant();
      else
        return Eigen3::EigenWrapper {std::forward<Arg>(arg)}.determinant();
      // Note: the global determinant function already handles TriangularView, DiagonalMatrix, and DiagonalWrapper
    }


    template<typename A, typename B>
    static constexpr auto
    sum(A&& a, B&& b)
    {
      auto s = make_self_contained<A, B>(std::forward<A>(a) + std::forward<B>(b));
      if constexpr ((dynamic_dimension<decltype(s), 0> and (not dynamic_dimension<A, 0> or not dynamic_dimension<B, 0>)) or
          (dynamic_dimension<decltype(s), 1> and (not dynamic_dimension<A, 1> or not dynamic_dimension<B, 1>)))
      {
        using S0 = index_descriptor_of_t<decltype(s), 0>; using S1 = index_descriptor_of_t<decltype(s), 1>;
        using A0 = index_descriptor_of_t<A, 0>; using A1 = index_descriptor_of_t<A, 1>;
        using B0 = index_descriptor_of_t<B, 0>; using B1 = index_descriptor_of_t<B, 1>;

        using R = std::conditional_t<dynamic_index_descriptor<S0>,
          std::conditional_t<dynamic_index_descriptor<A0>, std::conditional_t<dynamic_index_descriptor<B0>, S0, B0>, A0>, S0>;
        using C = std::conditional_t<dynamic_index_descriptor<S1>,
          std::conditional_t<dynamic_index_descriptor<A1>, std::conditional_t<dynamic_index_descriptor<B1>, S1, B1>, A1>, S1>;
        return internal::FixedSizeAdapter<std::decay_t<decltype(s)>, R, C> {std::move(s)};
      }
      else return s;
    }


    template<typename A, typename B>
    static constexpr auto
    contract(A&& a, B&& b)
    {
      if constexpr (diagonal_adapter<A>)
      {
        if constexpr (Eigen3::eigen_DiagonalWrapper<A> or Eigen3::eigen_DiagonalMatrix<A>)
          return make_self_contained<A, B>(std::forward<A>(a) * OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
        else
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<A>(a))).asDiagonal() *
            OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (diagonal_adapter<B>)
      {
        if constexpr (Eigen3::eigen_DiagonalWrapper<B> or Eigen3::eigen_DiagonalMatrix<B>)
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(std::forward<A>(a)) * std::forward<B>(b));
        else
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(std::forward<A>(a)) *
            OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<B>(b))).asDiagonal());
      }
      else if constexpr (triangular_adapter<A>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<A>(a))).template triangularView<uplo>() *
          OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (triangular_adapter<B>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        auto prod = OpenKalman::to_native_matrix<T>(std::forward<A>(a));
        prod.applyOnTheRight(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<B>(b))).template triangularView<uplo>());
        return prod;
      }
      else if constexpr (hermitian_adapter<A>)
      {
        constexpr auto uplo = hermitian_adapter<A, HermitianAdapterType::upper> ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<A>(a))).template selfadjointView<uplo>() *
          OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (hermitian_adapter<B>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        auto prod = OpenKalman::to_native_matrix<T>(std::forward<A>(a));
        prod.applyOnTheRight(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<B>(b))).template selfadjointView<uplo>());
        return prod;
      }
      else
      {
        return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(std::forward<A>(a)) *
          OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
    }


#ifdef __cpp_concepts
    template<bool on_the_right, writable A, indexible B> requires Eigen3::native_eigen_dense<A>
#else
    template<bool on_the_right, typename A, typename B, std::enable_if_t<writable<A> and Eigen3::native_eigen_dense<A>,int> = 0>
#endif
    static A&
    contract_in_place(A& a, B&& b)
    {
      decltype(auto) ma = [](A& a){
        if constexpr (Eigen3::native_eigen_array<A>) return a.matrix();
        else return (a);
      }(a);

      if constexpr (on_the_right)
        return ma.applyOnTheRight(OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      else
        return ma.applyOnTheLeft(OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      return a;
    }


    template<TriangleType triangle_type, typename A>
    static constexpr auto
    cholesky_factor(A&& a) noexcept
    {
      using NestedMatrix = std::decay_t<nested_matrix_of_t<A>>;
      using Scalar = scalar_type_of_t<A>;
      constexpr auto dim = index_dimension_of_v<A, 0>;
      using M = dense_writable_matrix_t<A, Scalar>;

      if constexpr (std::is_same_v<
        const NestedMatrix, const typename Eigen::MatrixBase<NestedMatrix>::ConstantReturnType>)
      {
        // If nested matrix is a positive constant matrix, construct the Cholesky factor using a shortcut.

        auto s = nested_matrix(std::forward<A>(a)).functor()();

        if (s < Scalar(0))
        {
          // Cholesky factor elements are complex, so throw an exception.
          throw (std::runtime_error("Cholesky_factor of constant SelfAdjointMatrix: covariance is indefinite"));
        }

        if constexpr(triangle_type == TriangleType::diagonal)
        {
          static_assert(diagonal_matrix<A>);
          auto vec = make_constant_matrix_like<A>(square_root(s), Dimensions<dim>{}, Dimensions<1>{});
          return DiagonalMatrix<decltype(vec)> {vec};
        }
        else if constexpr(triangle_type == TriangleType::lower)
        {
          auto col0 = make_constant_matrix_like<A>(square_root(s), Dimensions<dim>{}, Dimensions<1>{});
          auto othercols = make_zero_matrix_like<A>(get_index_descriptor<0>(a), get_index_descriptor<0>(a) - 1);
          return TriangularMatrix<M, triangle_type> {concatenate_horizontal(col0, othercols)};
        }
        else
        {
          static_assert(triangle_type == TriangleType::upper);
          auto row0 = make_constant_matrix_like<A>(square_root(s), Dimensions<1>{}, Dimensions<dim>{});
          auto otherrows = make_zero_matrix_like<A>(get_index_descriptor<0>(a) - 1, get_index_descriptor<0>(a));
          return TriangularMatrix<M, triangle_type> {concatenate_vertical(row0, otherrows)};
        }
      }
      else
      {
        // For the general case, perform an LLT Cholesky decomposition.
        M b;
        auto LL_x = a.view().llt();
        if (LL_x.info() == Eigen::Success)
        {
          if constexpr(triangle_type == hermitian_adapter_type_of_v<A>)
          {
            b = std::move(LL_x.matrixLLT());
          }
          else
          {
            constexpr unsigned int uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
            b.template triangularView<uplo>() = LL_x.matrixLLT().adjoint();
          }
        }
        else [[unlikely]]
        {
          // If covariance is not positive definite, use the more robust LDLT decomposition.
          auto LDL_x = nested_matrix(std::forward<A>(a)).ldlt();
          if ((not LDL_x.isPositive() and not LDL_x.isNegative()) or LDL_x.info() != Eigen::Success) [[unlikely]]
          {
            if (LDL_x.isPositive() and LDL_x.isNegative()) // Covariance is zero, even though decomposition failed.
            {
              if constexpr(triangle_type == TriangleType::lower)
                b.template triangularView<Eigen::Lower>() = make_zero_matrix_like(nested_matrix(a));
              else
                b.template triangularView<Eigen::Upper>() = make_zero_matrix_like(nested_matrix(a));
            }
            else // Covariance is indefinite, so throw an exception.
            {
              throw (std::runtime_error("Cholesky_factor of SelfAdjointMatrix: covariance is indefinite"));
            }
          }
          else if constexpr(triangle_type == TriangleType::lower)
          {
            b.template triangularView<Eigen::Lower>() =
              LDL_x.matrixL().toDenseMatrix() * LDL_x.vectorD().cwiseSqrt().asDiagonal();
          }
          else
          {
            b.template triangularView<Eigen::Upper>() =
              LDL_x.vectorD().cwiseSqrt().asDiagonal() * LDL_x.matrixU().toDenseMatrix();
          }
        }
        return TriangularMatrix<M, triangle_type> {std::move(b)};
      }
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
    {
      if constexpr (OpenKalman::Eigen3::eigen_ArrayWrapper<A>)
      {
        return rank_update_self_adjoint<significant_triangle>(nested_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
      else if constexpr (OpenKalman::Eigen3::native_eigen_array<A>)
      {
        return rank_update_self_adjoint<significant_triangle>(std::forward<A>(a).matrix(), std::forward<U>(u), alpha);
      }
      else
      {
        static_assert(writable<A>);
        constexpr auto s = significant_triangle == HermitianAdapterType::lower ? Eigen::Lower : Eigen::Upper;
        a.template selfadjointView<s>().template rankUpdate(std::forward<U>(u), alpha);
        return std::forward<A>(a);
      }
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      if constexpr (OpenKalman::Eigen3::eigen_ArrayWrapper<A>)
      {
        return rank_update_triangular<triangle>(nested_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
      else if constexpr (OpenKalman::Eigen3::native_eigen_array<A>)
      {
        return rank_update_triangular<triangle>(std::forward<A>(a).matrix(), std::forward<U>(u), alpha);
      }
      else
      {
        static_assert(writable<A>);
        constexpr auto t = triangle == TriangleType::lower ? Eigen::Lower : Eigen::Upper;
        using Scalar = scalar_type_of_t<A>;
        for (std::size_t i = 0; i < get_index_dimension_of<1>(u); i++)
        {
          if (Eigen::internal::llt_inplace<Scalar, t>::rankUpdate(a, get_column(u, i), alpha) >= 0)
            throw (std::runtime_error("rank_update_triangular: product is not positive definite"));
        }
        return std::forward<A>(a);
      }
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr auto
    solve(A&& a, B&& b)
    {
      using Scalar = scalar_type_of_t<A>;

      constexpr std::size_t a_rows = dynamic_rows<A> ? row_dimension_of_v<B> : row_dimension_of_v<A>;
      constexpr std::size_t a_cols = column_dimension_of_v<A>;
      constexpr std::size_t b_cols = column_dimension_of_v<B>;

      if constexpr (not Eigen3::native_eigen_matrix<A>)
      {
        decltype(auto) n = OpenKalman::to_native_matrix(std::forward<A>(a));
        static_assert(Eigen3::native_eigen_matrix<decltype(n)>);
        return solve<must_be_unique, must_be_exact>(std::forward<decltype(n)>(n), std::forward<B>(b));
      }
      else if constexpr (triangular_matrix<A>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(
          Eigen::Solve {std::forward<A>(a).template triangularView<uplo>(), std::forward<B>(b)});
      }
      else if constexpr (hermitian_matrix<A>)
      {
        constexpr auto uplo = hermitian_adapter_type_of_v<A> == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        auto v {std::forward<A>(a).template selfadjointView<uplo>()};
        auto llt {v.llt()};

        Eigen3::eigen_matrix_t<Scalar, a_cols, b_cols> ret;
        if (llt.info() == Eigen::Success)
        {
          ret = Eigen::Solve {llt, std::forward<B>(b)};
        }
        else [[unlikely]]
        {
          // A is semidefinite. Use LDLT decomposition instead.
          auto ldlt {v.ldlt()};
          if ((not ldlt.isPositive() and not ldlt.isNegative()) or ldlt.info() != Eigen::Success)
          {
            throw (std::runtime_error("Eigen solve (hermitian case): A is indefinite"));
          }
          ret = Eigen::Solve {ldlt, std::forward<B>(b)};
        }
        return ret;
      }
      else
      {
        if constexpr (must_be_exact or must_be_unique or true)
        {
          auto a_cols_rt = get_index_dimension_of<1>(a);
          Eigen::ColPivHouseholderQR<Eigen3::eigen_matrix_t<Scalar, a_rows, a_cols>> QR {std::forward<A>(a)};
          if constexpr (must_be_unique)
          {
            if (QR.rank() < a_cols_rt) throw std::runtime_error {"solve function requests a "
              "unique solution, but A is rank-deficient, so result X is not unique"};
          }

          auto res = QR.solve(std::forward<B>(b));

          if constexpr (must_be_exact)
          {
            bool a_solution_exists = (a*res).isApprox(b, a_cols_rt * std::numeric_limits<scalar_type_of_t<A>>::epsilon());

            if (a_solution_exists)
              return make_self_contained(std::move(res));
            else
              throw std::runtime_error {"solve function requests an exact solution, "
              "but the solution is only an approximation"};
          }
          else
          {
            return make_self_contained(std::move(res));
          }
        }
        else
        {
          Eigen::HouseholderQR<Eigen3::eigen_matrix_t<Scalar, a_rows, a_cols>> QR {std::forward<A>(a)};
          return make_self_contained(QR.solve(std::forward<B>(b)));
        }
      }
    }

  private:

      template<typename A>
      static constexpr auto
      QR_decomp_impl(A&& a)
      {
        using Scalar = scalar_type_of_t<A>;
        constexpr auto rows = row_dimension_of_v<A>;
        constexpr auto cols = column_dimension_of_v<A>;
        using MatrixType = Eigen3::eigen_matrix_t<Scalar, rows, cols>;
        using ResultType = Eigen3::eigen_matrix_t<Scalar, cols, cols>;

        Eigen::HouseholderQR<MatrixType> QR {std::forward<A>(a)};

        if constexpr (dynamic_columns<A>)
        {
          auto rt_cols = get_index_dimension_of<1>(a);

          ResultType ret {rt_cols, rt_cols};

          if constexpr (dynamic_rows<A>)
          {
            auto rt_rows = get_index_dimension_of<0>(a);

            if (rt_rows < rt_cols)
              ret << QR.matrixQR().topRows(rt_rows),
                Eigen3::eigen_matrix_t<Scalar, dynamic_size, dynamic_size>::Zero(rt_cols - rt_rows, rt_cols);
            else
              ret = QR.matrixQR().topRows(rt_cols);
          }
          else
          {
            if (rows < rt_cols)
              ret << QR.matrixQR().template topRows<rows>(),
                Eigen3::eigen_matrix_t<Scalar, dynamic_size, dynamic_size>::Zero(rt_cols - rows, rt_cols);
            else
              ret = QR.matrixQR().topRows(rt_cols);
          }

          return ret;
        }
        else
        {
          ResultType ret;

          if constexpr (dynamic_rows<A>)
          {
            auto rt_rows = get_index_dimension_of<0>(a);

            if (rt_rows < cols)
              ret << QR.matrixQR().topRows(rt_rows),
              Eigen3::eigen_matrix_t<Scalar, dynamic_size, dynamic_size>::Zero(cols - rt_rows, cols);
            else
              ret = QR.matrixQR().template topRows<cols>();
          }
          else
          {
            if constexpr (rows < cols)
              ret << QR.matrixQR().template topRows<rows>(), Eigen3::eigen_matrix_t<Scalar, cols - rows, cols>::Zero();
            else
              ret = QR.matrixQR().template topRows<cols>();
          }

          return ret;
        }
      }

  public:

    template<typename A>
    static constexpr auto
    LQ_decomposition(A&& a)
    {
      return make_triangular_matrix<TriangleType::lower>(make_self_contained(adjoint(QR_decomp_impl(adjoint(std::forward<A>(a))))));
    }


    template<typename A>
    static constexpr auto
    QR_decomposition(A&& a)
    {
      return make_triangular_matrix<TriangleType::upper>(QR_decomp_impl(std::forward<A>(a)));
    }

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_GENERAL_TRAITS_HPP
